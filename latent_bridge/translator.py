"""
RotAlignKVTranslator: the full pipeline in a single nn.Module.

Pipeline (per target layer l_t, paired with source layer l_s = pi(l_t)):

    K_s  --rotate (R_s)-->  K_s_rot  --W_K[l_t]-->  K_t_rot  --Q_b-->  codes
                                                                        |
                                                                        v
    K_t  <--(1-a)*K_t + a*K_hat--  K_hat  <--R_t^T--  K_t_rot_hat  <--Q_b^-1

All stages are composed in `translate_layer`. Calibration data is consumed
in `fit_from_pairs`, which fills in W_K and W_V via closed-form solvers.
"""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass, asdict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotation import make_rotation, fit_zca_whitening, apply_whitening, undo_whitening
from .procrustes import fit_alignment, alignment_quality
from .quantize import GaussianQuantizer


@dataclass
class TranslatorConfig:
    """Static configuration for the translator."""

    # Source model
    src_head_dim: int
    src_num_heads: int  # KV heads (under GQA, equals num_kv_heads not num_attention_heads)
    num_src_layers: int

    # Target model
    tgt_head_dim: int
    tgt_num_heads: int
    num_tgt_layers: int

    # Quantization
    quant_bits: int = 4

    # Rotation variant: 'orthogonal' (Haar-uniform), 'hadamard' (O(d log d)),
    # 'dct' (Fourier-family dense mixing), or 'identity' (negative control).
    rotation_kind: str = "orthogonal"

    # Whitening: if True, fit a per-layer ZCA whitening as a pre-processing
    # step before rotation + alignment. Corrects anisotropic scaling.
    use_whitening: bool = False
    # Canonicalize the target rotated coordinates too, then dewhiten after
    # projection. This gives us a symmetric quotient-space style alignment.
    use_target_whitening: bool = False
    # Restrict conditioning to a subset of streams. `kv` keeps the legacy
    # behavior, while `k` / `v` let us target the anchor or tail only.
    whitening_streams: str = "kv"
    target_whitening_streams: str = "kv"
    # Optional target-layer allowlist for conditioning. When unset, apply to
    # every target layer; when set, only the listed target layers use the
    # fitted whitening/dewhitening transforms.
    conditioning_target_layers: tuple[int, ...] | None = None

    # Alignment solver: 'auto' | 'identity' | 'procrustes'
    #                 | 'procrustes_rand' | 'ridge' | 'cca' | 'reduced_rank'
    #                 | 'grouped_' + any of the above
    #                 | 'grouped_transport' | 'grouped_permutation'
    #                 | 'grouped_signature_transport' | 'grouped_subspace_transport'
    #                 | 'grouped_canonical_transport' | 'grouped_adaptive_canonical_transport'
    #                 | 'grouped_covariance_transport'
    #                 | 'grouped_rotational_transport'
    #                 | 'grouped_fitted_rotation_transport'
    #                 | 'grouped_shared_basis_transport'
    #                 | 'grouped_template_transport'
    #                 | 'grouped_qk_retrieval_transport'
    #                 | 'grouped_contrastive_template_transport'
    #                 | 'grouped_template_subspace_transport'
    #                 | 'broadcast_template_transport'
    #                 | 'broadcast_template_ot_transport'
    #                 | 'broadcast_peak_template_ot_transport'
    #                 | 'broadcast_retrieval_spectrum_ot_transport'
    #                 | 'broadcast_qk_template_ot_transport'
    # Grouped variants fit one block per head-group instead of a single flat
    # all-head projection. When src/tgt head counts match, this degenerates to
    # true per-head alignment.
    alignment_method: str = "auto"
    ridge_lambda: float = 1e-3
    # Optional override for the closed-form source->target fit on selected
    # target layers/streams. This lets us test localized numerical fixes
    # without changing bridge-correction ridge terms or unrelated layers.
    fit_ridge_override_lambda: float | None = None
    fit_ridge_override_streams: str = "kv"
    fit_ridge_override_layers: tuple[int, ...] | None = None
    # When set with a fit ridge override, selected high-signal output channels
    # keep the base ridge while the remaining tail uses the override lambda.
    fit_ridge_protected_rank: int | None = None
    alignment_rank: int | None = None  # for cca / reduced_rank
    # Optional low-rank residual added on top of grouped soft transport.
    transport_residual_rank: int | None = None
    transport_temperature: float = 1.0
    transport_sinkhorn_iters: int = 8
    transport_signature_rank: int = 8
    transport_signature_weight: float = 0.0
    transport_template_bins: int = 64
    canonical_subspace_rank: int | None = None

    # Layer pairing: 'interp', 'cka', 'reverse', 'shifted', 'random', or a list
    # of length num_tgt_layers. The non-interp non-CKA modes are negative
    # controls for proving that layer structure matters.
    layer_pairing: str | list[int] = "interp"
    layer_selection_topk: int | None = None
    layer_selection_ratio: float = 1.0
    layer_selection_metric: str = "mean_cosine_similarity"

    # Optional head-group selection. When src/tgt head counts match this
    # degenerates to true per-head selection; otherwise the translator selects
    # aligned target head-groups defined by gcd(src_heads, tgt_heads).
    head_selection_topk: int | None = None
    head_selection_ratio: float = 1.0
    head_selection_metric: str = "mean_cosine_similarity"

    # Optional low-rank/shrinkage filter in the translated target space before
    # quantization. This is distinct from reduced-rank alignment: it denoises
    # the translated coordinates after source-to-target mapping.
    pre_quant_rank: int | None = None
    pre_quant_shrinkage: float = 0.0

    # Optional decoder-side correction after quantize/dequantize. `affine`
    # learns a diagonal scale+bias, `bridge_affine` learns a small coordinatewise
    # bridge over both the dequantized tensor and the pre-quant prediction,
    # `bridge_ridge` learns a full linear bridge over both signals,
    # `bridge_ridge_query` reuses that full bridge but gates it by live
    # target attention-template agreement, `bridge_low_rank_bank` fits a small
    # bank of low-rank bridge experts and mixes them online from the live
    # target attention profile, `bridge_ridge_residual_bank` keeps the global
    # bridge_ridge map and adds an attention-routed low-rank residual bank on
    # top, `bridge_ridge_qk_residual_bank` swaps that routing signal for a live
    # QK/retrieval profile, `bridge_ridge_qk_cab_bank` keeps the same QK-routed
    # bank structure but makes each expert a learned query-conditioned residual
    # adapter trained with prompt-local causal attention behavior,
    # `bridge_ridge_qk_predkl_bank` keeps that same routed bank structure but
    # supervises the experts with a calibration-time top-k next-token teacher,
    # and
    # `bridge_ridge_qk_weighted` keeps the global bridge
    # but fits it with calibration samples reweighted by target retrieval
    # importance, `bridge_ridge_qk_projector` feeds live query-conditioned
    # projector features directly into the bridge itself,
    # `bridge_ridge_qk_asym_projector` combines that full-rank query projector
    # with the shared-plus-private paired K/V interface so the projector itself
    # is modular rather than a plain residual patch, `bridge_ridge_qk_adapter`
    # adds a tiny learned low-rank query-conditioned residual on top of the
    # global bridge, `bridge_ridge_qk_affinity_adapter` adds a
    # query-conditioned affinity-matching objective to that same residual fit,
    # `bridge_ridge_qk_attnkl_adapter` adds sampled attention-logit KL on top
    # of the same residual fit, `bridge_ridge_qk_cab_adapter` swaps that
    # local teacher target for prompt-local causal attention-behavior
    # distillation inspired by CAB, `bridge_ridge_qk_emkd_adapter`
    # instead matches prompt-local token-interaction distributions inspired by
    # richer relational distillation, `bridge_ridge_qk_readout_adapter`
    # distills prompt-local attention readouts so the teacher is closer to
    # layer-level prediction behavior, `bridge_ridge_qk_predkl_adapter`
    # distills a top-k next-token teacher during calibration,
    # `bridge_ridge_qk_asym_adapter` upgrades the tiny residual bridge to a
    # shared-plus-private modular interface: one shared query-conditioned
    # bottleneck is fit jointly for K and V, then private K/V residual heads
    # sit on top of that shared bridge, and
    # `bridge_ridge_qk_asym_predkl_adapter` keeps that same
    # shared-plus-private interface but adds a prediction-level top-k
    # next-token teacher during calibration,
    # `bridge_ridge_qk_asym_dynmap_adapter` keeps the same
    # shared-plus-private interface but replaces static top-k KL with a
    # context-reweighted teacher over the top-k candidates, and
    # `bridge_ridge_qk_xattn_dynmap_adapter` keeps the explicit
    # query-conditioned cross-attention bridge interface from
    # `bridge_ridge_qk_xattn_adapter`, but adds that same
    # context-reweighted top-k teacher on top of the xattn bridge fit, and
    # `bridge_ridge_qk_module_adapter` upgrades that tiny xattn bridge into a
    # fuller attention-side transfer module with learned bridge slots and a
    # nonlinear readout, so the interface looks more like a small module
    # replacement than a local correction term,
    # `bridge_ridge_qk_module_replace` keeps that same slotted attention-side
    # module shape but trains it to predict the full corrected K/V directly
    # rather than only a residual on top of the fixed bridge, and
    # `bridge_ridge_qk_spanalign_module_replace` keeps that same direct-output
    # module shape but is intended to be fit from upstream token/span-remapped
    # calibration pairs rather than plain same-position token pairing, and
    # `bridge_ridge_qk_bytespan_module_replace` keeps that same direct-output
    # module shape but makes the upstream pairing tokenizer-agnostic by
    # aligning raw UTF-8 byte spans before fitting the bridge, and
    # `bridge_ridge_qk_ctxalign_module_replace` keeps that same direct-output
    # module shape but pushes the remapping upstream one step further: each
    # source token is fit against a small context-weighted mixture of target
    # tokens instead of a single hard-aligned target position, and
    # `bridge_ridge_qk_dynalign_ctxonly_module_replace` keeps the dynalign
    # candidate window but removes prediction-overlap scoring as a matched
    # teacher-null, and
    # `bridge_ridge_qk_dynalign_module_replace` keeps that same direct-output
    # module shape but upgrades the upstream remapping again: candidate target
    # tokens are scored by both local span/context agreement and prediction-side
    # token overlap, so calibration sees an output-aware token/span mixture
    # rather than only a context-weighted local mixture, and
    # `bridge_ridge_qk_dynalign_preserve_module_replace` keeps that same
    # output-aware dynalign remapping, but preserves a dominant target
    # subspace from the base bridge prediction and only asks the learned
    # module to repair the complementary tail, and
    # `bridge_ridge_qk_dynalign_eigenspace_module_replace` keeps the same
    # output-aware dynalign remapping but constrains the learned module to the
    # dominant target eigenspace itself, and
    # `bridge_ridge_qk_dynalign_saliency_module_replace` keeps the live
    # dynalign module shape but reweights the residual fit toward high-error,
    # query-relevant target dimensions, and
    # `bridge_ridge_qk_dynalign_saliency_preserve_module_replace` keeps the
    # live dynalign module shape but preserves the most salient target
    # directions from the base bridge and only repairs the learned-importance
    # tail, and
    # `bridge_ridge_qk_dynalign_anchor_tail_module_replace` keeps the same
    # saliency-selected preserve-versus-tail split, but only on the `V` side:
    # `K` stays on the live module-replacement path while `V` keeps a small
    # exact residual anchor against the base bridge and only the remaining
    # value-side tail is quantized, and
    # `bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace` narrows that
    # anchor-tail idea to layer-8 `V`: high-error/high-magnitude value
    # channels are escrowed to the base bridge during the module fit and kept
    # exact at runtime while the remaining value delta is quantized, and
    # `bridge_ridge_qk_dynalign_routed_module_replace` keeps that same direct
    # output-aware module shape but learns a tiny query-conditioned gate over
    # when to trust the repaired output versus the base bridge,
    # `bridge_ridge_qk_dynalign_value_routed_module_replace` keeps the winning
    # dynalign module path for `K` but makes `V` selective with that same tiny
    # query-conditioned gate,
    # `bridge_ridge_qk_dynalign_query_resampler_replace` keeps the same
    # slotted query-module replacement path but sanitizes bad fit tensors before
    # checkpoint materialization so failed seeds degrade to finite zero-sidecar
    # behavior instead of nonfinite `W_V`/module tensors,
    # `bridge_ridge_qk_dynalign_query_innovation_resampler_replace` reuses that
    # target-safe query resampler but fits only the source-conditioned innovation
    # over the target-side bridge prediction and adds a bounded residual at
    # fusion time,
    # `bridge_ridge_qk_dynalign_value_bank_module_replace` keeps that same
    # live `K` path but adds a tiny routed residual-expert bank only on the
    # `V` side,
    # `bridge_ridge_qk_dynalign_value_query_bank_module_replace` keeps the
    # same live `K` path and module-replacement `V` path, but routes a
    # residual expert bank directly from clustered query features instead of
    # a runtime attention-template profile,
    # `bridge_ridge_qk_dynalign_value_routed_bank_module_replace` keeps the
    # surviving value-routed blend and only adds a sparse top-2 bank correction
    # on top of that `V` path,
    # `bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace` keeps
    # that same live value-routed path, but adds a learned value-only residual
    # sidecar that is accepted only when a second query-conditioned verifier
    # gate predicts the sidecar beats the routed baseline, and
    # `bridge_ridge_qk_dynalign_dwakd_module_replace` keeps that same dynalign
    # upstream remapping but adds DWA-KD-style confidence weighting and a
    # stronger dynamic prediction teacher during module fitting, and
    # `bridge_ridge_qk_dynalign_likelihood_module_replace` keeps that weighted
    # dynalign teacher but injects empirical target next-token likelihood mass
    # into the aligned top-k teacher, and
    # `bridge_ridge_qk_dynalign_spanalm_module_replace` keeps dynalign and
    # blends a small span-window approximate-likelihood teacher instead of a
    # hard observed-next-token boost, and
    # `bridge_ridge_qk_dynalign_prefdist_module_replace` keeps dynalign and
    # adds pairwise preference distillation over aligned target output rows, and
    # `bridge_ridge_qk_dynalign_dwainteract_module_replace` stacks the DWA
    # confidence/dynamic teacher with prompt-local interaction distillation, and
    # `bridge_ridge_qk_dynalign_interact_module_replace` keeps that same
    # dynalign remapping but adds prompt-local interaction distillation during
    # module fitting, and
    # `bridge_ridge_qk_dpalign_module_replace` keeps that same direct-output
    # module shape but switches from local candidate mixtures to a global
    # monotone dynamic-program alignment over context-plus-output scores before
    # fitting the bridge, and
    # `bridge_ridge_qk_tokenbasis_replace` keeps the same slotted
    # attention-side module shape but constrains the direct-output replacement
    # to a basis distilled from the target next-token output rows, and
    # `bridge_ridge_qk_sae_adapter` swaps the dense shared bottleneck for a
    # sparse shared codebook: paired K/V query-conditioned signals produce a
    # small top-k latent code that is decoded separately for K and V, and
    # `bridge_ridge_qk_generated_adapter` upgrades that to a continuous
    # instance-specific bridge by generating per-sample mixture weights over a
    # shared bank of low-rank bridge atoms from paired K/V query features;
    # `ridge` learns a
    # full linear map plus bias
    # in rotated
    # target space, and `low_rank` learns a reduced-rank linear map plus bias
    # as a tiny bridge adapter.
    quantization_correction: str = "none"
    quantization_correction_rank: int | None = None
    bridge_bank_size: int = 4
    bridge_bank_temperature: float = 8.0

    # Fusion rule for combining target and translated K/V. 'static' keeps the
    # checkpointed scalar gates as-is; cosine-based rules attenuate translated
    # KV when its flattened direction disagrees with the target cache.
    fusion_rule: str = "static"
    learned_fusion_dropout: float = 0.0

    # Rotation RNG
    seed: int = 0

    def __post_init__(self) -> None:
        self.bridge_bank_size = int(self.bridge_bank_size)
        if self.bridge_bank_size < 0:
            raise ValueError(f"bridge_bank_size must be non-negative, got {self.bridge_bank_size}")
        valid_streams = {"kv", "k", "v"}
        if self.whitening_streams not in valid_streams:
            raise ValueError(f"Invalid whitening_streams: {self.whitening_streams}")
        if self.target_whitening_streams not in valid_streams:
            raise ValueError(f"Invalid target_whitening_streams: {self.target_whitening_streams}")
        if self.fit_ridge_override_streams not in valid_streams:
            raise ValueError(f"Invalid fit_ridge_override_streams: {self.fit_ridge_override_streams}")
        if self.fit_ridge_override_lambda is not None:
            override_lambda = float(self.fit_ridge_override_lambda)
            if not math.isfinite(override_lambda) or override_lambda <= 0.0:
                raise ValueError(
                    "fit_ridge_override_lambda must be finite and positive "
                    f"when set, got {self.fit_ridge_override_lambda}"
                )
            self.fit_ridge_override_lambda = override_lambda
        if self.conditioning_target_layers is not None:
            self.conditioning_target_layers = tuple(
                sorted({int(layer) for layer in self.conditioning_target_layers})
            )
        if self.fit_ridge_override_layers is not None:
            self.fit_ridge_override_layers = tuple(
                sorted({int(layer) for layer in self.fit_ridge_override_layers})
            )
        if self.fit_ridge_protected_rank is not None:
            self.fit_ridge_protected_rank = int(self.fit_ridge_protected_rank)
            if self.fit_ridge_protected_rank <= 0:
                raise ValueError(
                    "fit_ridge_protected_rank must be positive when set, "
                    f"got {self.fit_ridge_protected_rank}"
                )


class RotAlignKVTranslator(nn.Module):
    """Cross-model KV-cache translator with rotational alignment and optional
    Lloyd-Max quantization.

    Parameters that are *learned from data*:
      * W_K[l], W_V[l] : linear projections fit by Procrustes/ridge in closed form
      * alpha_K[l], alpha_V[l] : scalar fusion gates, trainable with grad descent
                                 or fit via 1D line search

    Parameters that are *fixed*:
      * R_s, R_t : random orthogonal matrices (Gaussianization rotations)
      * quantizer.codebook : Lloyd-Max codebook for N(0, 1)
    """

    def __init__(self, config: TranslatorConfig):
        super().__init__()
        self.config = config

        # --- Fixed rotations (random orthogonal or randomized Hadamard) ---
        R_s = make_rotation(config.src_head_dim, kind=config.rotation_kind, seed=config.seed)
        R_t = make_rotation(config.tgt_head_dim, kind=config.rotation_kind, seed=config.seed + 1)
        self.register_buffer("R_s", R_s)
        self.register_buffer("R_t", R_t)

        # Flattened per-layer dimensions after rotation
        self.d_s = config.src_num_heads * config.src_head_dim
        self.d_t = config.tgt_num_heads * config.tgt_head_dim

        # --- Per-layer linear alignments (K and V get separate Ws) ---
        # Initialized to zero; filled in by fit_from_pairs before use.
        self.W_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_s, self.d_t)) for _ in range(config.num_tgt_layers)]
        )
        self.W_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_s, self.d_t)) for _ in range(config.num_tgt_layers)]
        )

        # --- Optional per-layer ZCA whitening buffers ---
        # Lazily allocated during fit_from_pairs if use_whitening=True.
        if config.use_whitening:
            self.whiten_K_src = nn.ParameterList(
                [nn.Parameter(torch.eye(self.d_s), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
            self.whiten_V_src = nn.ParameterList(
                [nn.Parameter(torch.eye(self.d_s), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
            self.whiten_K_mean = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.d_s), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
            self.whiten_V_mean = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.d_s), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
        else:
            self.whiten_K_src = None
            self.whiten_V_src = None
            self.whiten_K_mean = None
            self.whiten_V_mean = None

        if config.use_target_whitening:
            self.whiten_K_tgt = nn.ParameterList(
                [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
            self.whiten_V_tgt = nn.ParameterList(
                [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
            self.whiten_K_tgt_inv = nn.ParameterList(
                [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
            self.whiten_V_tgt_inv = nn.ParameterList(
                [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
            self.whiten_K_tgt_mean = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
            self.whiten_V_tgt_mean = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
            )
        else:
            self.whiten_K_tgt = None
            self.whiten_V_tgt = None
            self.whiten_K_tgt_inv = None
            self.whiten_V_tgt_inv = None
            self.whiten_K_tgt_mean = None
            self.whiten_V_tgt_mean = None

        # --- Per-layer scalar fusion gates (stored as logits; sigmoid at use) ---
        # Start at 0 -> sigmoid(0) = 0.5, equal weight on own vs translated.
        self.gate_K = nn.Parameter(torch.zeros(config.num_tgt_layers))
        self.gate_V = nn.Parameter(torch.zeros(config.num_tgt_layers))

        # --- Lloyd-Max quantizer (fixed codebook) ---
        self.quantizer = GaussianQuantizer(bits=config.quant_bits)

        # --- Layer pairing map ---
        self.layer_map = self._build_layer_map(config)
        self.register_buffer(
            "layer_selected_mask",
            torch.ones(config.num_tgt_layers, dtype=torch.bool),
        )
        self.register_buffer(
            "head_selected_mask",
            torch.ones(config.num_tgt_layers, config.tgt_num_heads, dtype=torch.bool),
        )
        group_count, _, _ = self._head_group_layout()
        self.register_buffer(
            "transport_plan_K",
            torch.zeros(config.num_tgt_layers, group_count, group_count),
        )
        self.register_buffer(
            "transport_plan_V",
            torch.zeros(config.num_tgt_layers, group_count, group_count),
        )
        # Calibration-time head descriptors are used only while fitting
        # template/signature-based transport and are not checkpoint state.
        self._transport_src_group_templates: list[torch.Tensor] | None = None

        self._transport_tgt_group_templates: list[torch.Tensor] | None = None
        self._transport_src_group_template_banks: list[torch.Tensor] | None = None
        self._transport_tgt_group_template_banks: list[torch.Tensor] | None = None
        self._broadcast_transport_plan_K: list[torch.Tensor] | None = None
        self._broadcast_transport_plan_V: list[torch.Tensor] | None = None
        self._bridge_prompt_cluster_labels: list[torch.Tensor] | None = None
        self._bridge_sample_prompt_ids: torch.Tensor | None = None
        self._bridge_sample_weights: list[torch.Tensor] | None = None
        self._bridge_sample_query_features: list[torch.Tensor] | None = None
        self._bridge_prediction_teacher_log_probs: torch.Tensor | None = None
        self._bridge_prediction_teacher_output_rows: torch.Tensor | None = None
        self.register_buffer(
            "bridge_runtime_templates",
            torch.zeros(config.num_tgt_layers, config.transport_template_bins, dtype=torch.float32),
        )
        self.register_buffer(
            "bridge_bank_templates",
            torch.zeros(
                config.num_tgt_layers,
                config.bridge_bank_size,
                config.transport_template_bins,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "bridge_bank_query_centroids",
            torch.zeros(
                config.num_tgt_layers,
                config.bridge_bank_size,
                self.d_t,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "bridge_bank_priors",
            torch.zeros(config.num_tgt_layers, config.bridge_bank_size, dtype=torch.float32),
        )

        # --- Optional pre-quant denoising filters and quantization repair ---
        bridge_rank = max(1, int(config.quantization_correction_rank or 8))
        self.pre_quant_filter_K = nn.ParameterList(
            [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.pre_quant_filter_V = nn.ParameterList(
            [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_scale_K = nn.ParameterList(
            [nn.Parameter(torch.ones(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_scale_V = nn.ParameterList(
            [nn.Parameter(torch.ones(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_aux_scale_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_aux_scale_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_proj_K = nn.ParameterList(
            [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_proj_V = nn.ParameterList(
            [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_aux_proj_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_aux_proj_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_proj_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_proj_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_aux_proj_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_aux_proj_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_resid_K_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_resid_K_right = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_aux_resid_K_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_aux_resid_K_right = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_resid_V_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_resid_V_right = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_aux_resid_V_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_aux_resid_V_right = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_shared_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_shared_aux_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_shared_K_right = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_shared_V_right = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_sparse_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_sparse_aux_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_sparse_K_right = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_sparse_V_right = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_hyper_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, config.bridge_bank_size), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_hyper_aux_left = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, config.bridge_bank_size), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_xattn_q = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_xattn_k = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_xattn_v = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_xattn_K_out = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_xattn_V_out = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_module_slots = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.bridge_bank_size, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_module_q = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_module_k = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_module_v = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_module_hidden = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_module_K_out = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_module_V_out = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_route_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, 1), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_route_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, 1), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_route_K_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(1), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_route_V_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(1), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_sidecar_route_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, 1), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_sidecar_route_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, 1), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_sidecar_route_K_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(1), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_sidecar_route_V_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(1), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_token_basis = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_token_K_coeff = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_query_token_V_coeff = nn.ParameterList(
            [nn.Parameter(torch.zeros(bridge_rank, bridge_rank), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_preserve_proj_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_preserve_proj_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.d_t, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_bias_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_bias_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.bridge_bank_proj_K_left = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, self.d_t, bridge_rank),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_proj_K_right = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, bridge_rank, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_aux_proj_K_left = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, self.d_t, bridge_rank),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_aux_proj_K_right = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, bridge_rank, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_bias_K = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, 1, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_query_resid_K_left = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, self.d_t, bridge_rank),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_query_resid_K_right = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, bridge_rank, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_query_aux_resid_K_left = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, self.d_t, bridge_rank),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_query_aux_resid_K_right = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, bridge_rank, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_proj_V_left = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, self.d_t, bridge_rank),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_query_resid_V_left = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, self.d_t, bridge_rank),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_query_resid_V_right = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, bridge_rank, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_query_aux_resid_V_left = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, self.d_t, bridge_rank),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_query_aux_resid_V_right = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, bridge_rank, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_proj_V_right = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, bridge_rank, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_aux_proj_V_left = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, self.d_t, bridge_rank),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_aux_proj_V_right = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, bridge_rank, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.bridge_bank_bias_V = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(config.bridge_bank_size, 1, self.d_t),
                    requires_grad=False,
                )
                for _ in range(config.num_tgt_layers)
            ]
        )
        self.fusion_src_scale_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.fusion_src_scale_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.fusion_tgt_scale_K = nn.ParameterList(
            [nn.Parameter(torch.ones(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.fusion_tgt_scale_V = nn.ParameterList(
            [nn.Parameter(torch.ones(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.fusion_bias_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.fusion_bias_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        if config.learned_fusion_dropout > 0.0:
            self.fusion_head_proj_K = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(config.tgt_num_heads, 2 * config.tgt_head_dim, config.tgt_head_dim),
                        requires_grad=False,
                    )
                    for _ in range(config.num_tgt_layers)
                ]
            )
            self.fusion_head_proj_V = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(config.tgt_num_heads, 2 * config.tgt_head_dim, config.tgt_head_dim),
                        requires_grad=False,
                    )
                    for _ in range(config.num_tgt_layers)
                ]
            )
            self.fusion_head_bias_K = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(config.tgt_num_heads, config.tgt_head_dim),
                        requires_grad=False,
                    )
                    for _ in range(config.num_tgt_layers)
                ]
            )
            self.fusion_head_bias_V = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(config.tgt_num_heads, config.tgt_head_dim),
                        requires_grad=False,
                    )
                    for _ in range(config.num_tgt_layers)
                ]
            )
        else:
            self.fusion_head_proj_K = None
            self.fusion_head_proj_V = None
            self.fusion_head_bias_K = None
            self.fusion_head_bias_V = None
        self._fitted = False

    def _conditioning_layer_applies(self, tgt_layer_idx: int) -> bool:
        layers = self.config.conditioning_target_layers
        return layers is None or int(tgt_layer_idx) in layers

    @staticmethod
    def _conditioning_stream_applies(stream_spec: str, kind: str) -> bool:
        return stream_spec == "kv" or stream_spec == kind.lower()

    def _source_whitening_applies(self, tgt_layer_idx: int, kind: str) -> bool:
        return (
            self.config.use_whitening
            and self._conditioning_layer_applies(tgt_layer_idx)
            and self._conditioning_stream_applies(self.config.whitening_streams, kind)
        )

    def _target_whitening_applies(self, tgt_layer_idx: int, kind: str) -> bool:
        return (
            self.config.use_target_whitening
            and self._conditioning_layer_applies(tgt_layer_idx)
            and self._conditioning_stream_applies(self.config.target_whitening_streams, kind)
        )

    def _fit_ridge_override_applies(self, tgt_layer_idx: int, kind: str) -> bool:
        layers = self.config.fit_ridge_override_layers
        return (
            self.config.fit_ridge_override_lambda is not None
            and (layers is None or int(tgt_layer_idx) in layers)
            and self._conditioning_stream_applies(self.config.fit_ridge_override_streams, kind)
        )

    def _fit_ridge_lambda(self, tgt_layer_idx: int, kind: str) -> float:
        if self._fit_ridge_override_applies(tgt_layer_idx, kind):
            return float(self.config.fit_ridge_override_lambda)
        return float(self.config.ridge_lambda)

    @staticmethod
    def _fit_ridge_top_output_mask(target: torch.Tensor, rank: int) -> torch.Tensor:
        target_f = target.float()
        score = target_f.abs().amax(dim=0) + target_f.std(dim=0, unbiased=False)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        keep = max(1, min(int(rank), score.numel()))
        top_idx = torch.topk(score, k=keep, largest=True).indices
        mask = torch.zeros(score.numel(), dtype=torch.bool, device=target.device)
        mask[top_idx] = True
        return mask

    def _fit_ridge_protected_output_mask(
        self,
        target: torch.Tensor,
        tgt_layer_idx: int,
        kind: str,
    ) -> torch.Tensor | None:
        rank = self.config.fit_ridge_protected_rank
        if rank is None or not self._fit_ridge_override_applies(tgt_layer_idx, kind):
            return None
        return self._fit_ridge_top_output_mask(target, int(rank))

    @staticmethod
    def _clip_rank_for_outputs(rank: int | None, x_dim: int, y_dim: int) -> int | None:
        if rank is None:
            return None
        return max(1, min(int(rank), int(x_dim), int(y_dim)))

    @staticmethod
    def _guard_query_resampler_fit_tensor(
        tensor: torch.Tensor,
        fallback: torch.Tensor,
        *,
        max_abs: float = 1.0e6,
    ) -> torch.Tensor:
        stats_tensor = tensor.detach().float()
        if stats_tensor.numel() == 0:
            return tensor
        finite = bool(torch.isfinite(stats_tensor).all().item())
        max_value = float(stats_tensor.abs().max().item()) if finite else math.inf
        if finite and math.isfinite(max_value) and max_value <= float(max_abs):
            return tensor
        return fallback.to(device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _bound_query_innovation_delta(
        delta: torch.Tensor,
        reference: torch.Tensor,
        *,
        max_norm_ratio: float = 0.25,
    ) -> torch.Tensor:
        delta_norm = delta.float().norm(dim=-1, keepdim=True)
        reference_norm = reference.float().norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        safe_delta_norm = delta_norm.clamp_min(1.0e-6)
        max_ratio = max(float(max_norm_ratio), 0.0)
        scale = torch.minimum(
            torch.ones_like(delta_norm),
            (max_ratio * reference_norm) / safe_delta_norm,
        )
        scale = torch.nan_to_num(scale, nan=0.0, posinf=1.0, neginf=0.0)
        return delta * scale.to(device=delta.device, dtype=delta.dtype)

    def _fit_alignment_with_protected_outputs(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        method: str,
        lam: float,
        rank: int | None = None,
        protected_output_mask: torch.Tensor | None = None,
        protected_lam: float | None = None,
    ) -> torch.Tensor:
        if protected_output_mask is None or protected_lam is None:
            return fit_alignment(X, Y, method=method, lam=lam, rank=rank)
        mask = protected_output_mask.to(device=Y.device, dtype=torch.bool)
        if mask.numel() != Y.shape[1]:
            raise ValueError(
                "protected_output_mask must match target feature dimension, "
                f"got {mask.numel()} and {Y.shape[1]}"
            )
        if not bool(mask.any()):
            return fit_alignment(X, Y, method=method, lam=lam, rank=rank)
        if bool(mask.all()):
            return fit_alignment(X, Y, method=method, lam=protected_lam, rank=rank)

        W = fit_alignment(X, Y, method=method, lam=lam, rank=rank)
        protected_cols = torch.nonzero(mask, as_tuple=False).flatten()
        protected_rank = self._clip_rank_for_outputs(rank, X.shape[1], int(protected_cols.numel()))
        W_protected = fit_alignment(
            X,
            Y[:, protected_cols],
            method=method,
            lam=float(protected_lam),
            rank=protected_rank,
        )
        W[:, protected_cols] = W_protected.to(dtype=W.dtype, device=W.device)
        return W

    @staticmethod
    def _build_layer_map(cfg: TranslatorConfig) -> list[int]:
        if isinstance(cfg.layer_pairing, list):
            assert len(cfg.layer_pairing) == cfg.num_tgt_layers
            return list(cfg.layer_pairing)
        interp = [
            min(int(round(i * cfg.num_src_layers / cfg.num_tgt_layers)), cfg.num_src_layers - 1)
            for i in range(cfg.num_tgt_layers)
        ]
        if cfg.layer_pairing in {"interp", "cka"}:
            return interp
        if cfg.layer_pairing == "reverse":
            return [cfg.num_src_layers - 1 - idx for idx in interp]
        if cfg.layer_pairing == "shifted":
            return [(idx + 1) % cfg.num_src_layers for idx in interp]
        if cfg.layer_pairing == "random":
            gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
            return torch.randint(
                low=0,
                high=cfg.num_src_layers,
                size=(cfg.num_tgt_layers,),
                generator=gen,
            ).tolist()
        raise ValueError(f"Unknown layer_pairing: {cfg.layer_pairing}")

    @staticmethod
    def _linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
        """Linear CKA similarity used for SemAlign-style layer pairing."""
        Xc = X.float() - X.float().mean(dim=0, keepdim=True)
        Yc = Y.float() - Y.float().mean(dim=0, keepdim=True)
        xy = Yc.T @ Xc
        xx = Xc.T @ Xc
        yy = Yc.T @ Yc
        num = float((xy * xy).sum())
        den = float(torch.sqrt((xx * xx).sum() * (yy * yy).sum()).clamp_min(eps))
        return num / den

    def _fit_cka_layer_map(
        self,
        src_kvs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        tgt_kvs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[int]:
        """Pair each target layer with the most CKA-similar source layer."""
        src_feats: list[torch.Tensor] = []
        tgt_feats: list[torch.Tensor] = []

        for K_s, _ in src_kvs:
            src_feats.append(self._rotate_and_flatten(K_s, self.R_s).reshape(-1, self.d_s))
        for K_t, _ in tgt_kvs:
            tgt_feats.append(self._rotate_and_flatten(K_t, self.R_t).reshape(-1, self.d_t))

        used: set[int] = set()
        layer_map: list[int] = []
        for tgt_idx, tgt_feat in enumerate(tgt_feats):
            best_src = 0
            best_score = float("-inf")
            for src_idx, src_feat in enumerate(src_feats):
                if src_idx in used and len(src_feats) >= len(tgt_feats):
                    continue
                score = self._linear_cka(src_feat, tgt_feat)
                if score > best_score:
                    best_score = score
                    best_src = src_idx
            layer_map.append(best_src)
            used.add(best_src)
        return layer_map

    def _apply_layer_selection(self, diagnostics: dict[int, dict]) -> None:
        """Choose which target layers receive translated communication."""
        n_layers = self.config.num_tgt_layers
        scores: list[tuple[float, int]] = []
        for tgt_idx, diag in diagnostics.items():
            if self.config.layer_selection_metric == "mean_cosine_similarity":
                score = 0.5 * (
                    diag["K"]["mean_cosine_similarity"] + diag["V"]["mean_cosine_similarity"]
                )
            elif self.config.layer_selection_metric == "negative_error":
                score = -0.5 * (
                    diag["K"]["relative_frobenius_error"] + diag["V"]["relative_frobenius_error"]
                )
            else:
                raise ValueError(
                    f"Unknown layer_selection_metric: {self.config.layer_selection_metric}"
                )
            scores.append((float(score), tgt_idx))
        scores.sort(reverse=True)

        if self.config.layer_selection_topk is not None:
            keep = max(1, min(n_layers, self.config.layer_selection_topk))
        else:
            keep = max(1, min(n_layers, int(round(n_layers * self.config.layer_selection_ratio))))

        mask = torch.zeros(n_layers, dtype=torch.bool, device=self.layer_selected_mask.device)
        for _, tgt_idx in scores[:keep]:
            mask[tgt_idx] = True
        self.layer_selected_mask.copy_(mask)

    def selected_layer_indices(self) -> list[int]:
        return [
            idx for idx, selected in enumerate(self.layer_selected_mask.tolist()) if selected
        ]

    def is_layer_selected(self, tgt_layer_idx: int) -> bool:
        return bool(self.layer_selected_mask[tgt_layer_idx].item())

    def selected_head_count(self, tgt_layer_idx: int) -> int:
        return int(self.head_selected_mask[tgt_layer_idx].sum().item())

    def is_head_selected(self, tgt_layer_idx: int, head_idx: int) -> bool:
        return bool(self.head_selected_mask[tgt_layer_idx, head_idx].item())

    def apply_head_selection(
        self,
        kv: torch.Tensor,
        tgt_layer_idx: int,
        *,
        fill: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self.head_selected_mask[tgt_layer_idx].view(1, -1, 1, 1).to(device=kv.device)
        if fill is None:
            fill = torch.zeros_like(kv)
        return torch.where(mask, kv, fill)

    def set_fixed_gates(self, alpha_k: float, alpha_v: float | None = None) -> None:
        """Overwrite all gates with a fixed scalar in [0, 1]."""
        alpha_v = alpha_k if alpha_v is None else alpha_v
        eps = 1e-6
        a_k = min(max(alpha_k, eps), 1.0 - eps)
        a_v = min(max(alpha_v, eps), 1.0 - eps)
        logit_k = torch.logit(torch.tensor(a_k, dtype=self.gate_K.dtype, device=self.gate_K.device))
        logit_v = torch.logit(torch.tensor(a_v, dtype=self.gate_V.dtype, device=self.gate_V.device))
        self.gate_K.data.fill_(float(logit_k))
        self.gate_V.data.fill_(float(logit_v))

    def set_layer_gates(self, tgt_layer_idx: int, alpha_k: float | None = None, alpha_v: float | None = None) -> None:
        """Update one layer's gates without touching the rest of the stack."""
        eps = 1e-6
        if alpha_k is not None:
            a_k = min(max(alpha_k, eps), 1.0 - eps)
            logit_k = torch.logit(torch.tensor(a_k, dtype=self.gate_K.dtype, device=self.gate_K.device))
            self.gate_K.data[tgt_layer_idx] = logit_k
        if alpha_v is not None:
            a_v = min(max(alpha_v, eps), 1.0 - eps)
            logit_v = torch.logit(torch.tensor(a_v, dtype=self.gate_V.dtype, device=self.gate_V.device))
            self.gate_V.data[tgt_layer_idx] = logit_v

    def gate_value(self, tgt_layer_idx: int) -> tuple[float, float]:
        return (
            float(torch.sigmoid(self.gate_K[tgt_layer_idx]).detach()),
            float(torch.sigmoid(self.gate_V[tgt_layer_idx]).detach()),
        )

    def gate_values(self) -> list[tuple[float, float]]:
        return [self.gate_value(idx) for idx in range(self.config.num_tgt_layers)]

    @staticmethod
    def _flatten_cosine_similarity(target_kv: torch.Tensor, translated_kv: torch.Tensor) -> torch.Tensor:
        target_flat = target_kv.float().reshape(target_kv.shape[0], -1)
        translated_flat = translated_kv.float().reshape(translated_kv.shape[0], -1)
        return torch.cosine_similarity(target_flat, translated_flat, dim=-1, eps=1e-8).mean()

    @staticmethod
    def _tokenwise_cosine_similarity(target_kv: torch.Tensor, translated_kv: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = target_kv.shape
        target_flat = target_kv.float().permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        translated_flat = translated_kv.float().permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        cosine = torch.cosine_similarity(target_flat, translated_flat, dim=-1, eps=1e-8)
        return cosine.unsqueeze(1).unsqueeze(-1)

    @staticmethod
    def _tokenwise_mean_square(x: torch.Tensor) -> torch.Tensor:
        return x.float().pow(2).mean(dim=(1, 3), keepdim=True)

    def _effective_gate(
        self,
        base_gate: torch.Tensor,
        target_kv: torch.Tensor,
        translated_kv: torch.Tensor,
        fusion_rule: str,
    ) -> torch.Tensor:
        if fusion_rule == "static":
            return base_gate

        residual = (translated_kv - target_kv).float()
        tokenwise = fusion_rule.endswith("_tokenwise")
        base_rule = fusion_rule.removesuffix("_tokenwise")

        if tokenwise:
            residual_var = self._tokenwise_mean_square(residual)
            target_var = self._tokenwise_mean_square(target_kv).clamp_min(1e-8)
            translated_var = self._tokenwise_mean_square(translated_kv).clamp_min(1e-8)
            cosine = self._tokenwise_cosine_similarity(target_kv, translated_kv)
        else:
            residual_var = residual.pow(2).mean()
            target_var = target_kv.float().pow(2).mean().clamp_min(1e-8)
            translated_var = translated_kv.float().pow(2).mean().clamp_min(1e-8)
            cosine = self._flatten_cosine_similarity(target_kv, translated_kv)

        if base_rule == "cosine":
            scale = cosine.clamp(0.0, 1.0)
        elif base_rule == "cosine_shifted":
            scale = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
        elif base_rule == "js_shrinkage":
            scale = (1.0 - residual_var / translated_var).clamp(0.0, 1.0)
        elif base_rule == "kalman":
            scale = (target_var / (target_var + residual_var)).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown fusion_rule: {fusion_rule}")
        return base_gate * scale.to(device=base_gate.device, dtype=base_gate.dtype)

    # ------------------------------------------------------------------
    # Core geometric operations
    # ------------------------------------------------------------------

    def _rotate_and_flatten(
        self, kv: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """Rotate along head_dim, then flatten (num_heads, head_dim) -> (num_heads*head_dim).

        Args:
            kv: [batch, num_heads, seq, head_dim]
            R:  [head_dim, head_dim]
        Returns:
            [batch, seq, num_heads * head_dim]
        """
        rotated = kv @ R  # [b, h, s, d]
        b, h, s, d = rotated.shape
        # [b, h, s, d] -> [b, s, h, d] -> [b, s, h*d]
        return rotated.transpose(1, 2).contiguous().view(b, s, h * d)

    @staticmethod
    def _flatten_without_rotation(kv: torch.Tensor) -> torch.Tensor:
        b, h, s, d = kv.shape
        return kv.transpose(1, 2).contiguous().view(b, s, h * d)

    def _unflatten_and_inverse_rotate(
        self,
        kv_flat: torch.Tensor,
        num_heads: int,
        head_dim: int,
        R_T: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse of _rotate_and_flatten."""
        b, s, hd = kv_flat.shape
        assert hd == num_heads * head_dim
        kv = kv_flat.view(b, s, num_heads, head_dim).transpose(1, 2).contiguous()
        return kv @ R_T

    def _head_group_layout(self) -> tuple[int, int, int]:
        group_count = max(1, math.gcd(self.config.src_num_heads, self.config.tgt_num_heads))
        return (
            group_count,
            self.config.src_num_heads // group_count,
            self.config.tgt_num_heads // group_count,
        )

    def _group_feature_slices(
        self,
        *,
        use_target: bool,
    ) -> list[slice]:
        group_count, src_heads_per_group, tgt_heads_per_group = self._head_group_layout()
        heads_per_group = tgt_heads_per_group if use_target else src_heads_per_group
        head_dim = self.config.tgt_head_dim if use_target else self.config.src_head_dim
        group_dim = heads_per_group * head_dim
        return [
            slice(group_idx * group_dim, (group_idx + 1) * group_dim)
            for group_idx in range(group_count)
        ]

    def _head_feature_slices(
        self,
        *,
        use_target: bool,
    ) -> list[slice]:
        head_count = self.config.tgt_num_heads if use_target else self.config.src_num_heads
        head_dim = self.config.tgt_head_dim if use_target else self.config.src_head_dim
        return [slice(head_idx * head_dim, (head_idx + 1) * head_dim) for head_idx in range(head_count)]

    def _fit_grouped_whitening(
        self,
        X: torch.Tensor,
        *,
        use_target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = self.d_t if use_target else self.d_s
        W = torch.zeros(dim, dim, dtype=X.dtype, device=X.device)
        mean = torch.zeros(1, dim, dtype=X.dtype, device=X.device)
        for feature_slice in self._group_feature_slices(use_target=use_target):
            W_group, mean_group = fit_zca_whitening(X[:, feature_slice])
            W[feature_slice, feature_slice] = W_group.to(dtype=X.dtype)
            mean[:, feature_slice] = mean_group.to(dtype=X.dtype)
        return W, mean

    def _fit_grouped_alignment(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        method: str,
        lam: float,
        rank: int | None,
    ) -> torch.Tensor:
        base_method = method.removeprefix("grouped_")
        if not base_method:
            raise ValueError("Grouped alignment method must specify a base solver")
        W = torch.zeros(self.d_s, self.d_t, dtype=X.dtype, device=X.device)
        src_slices = self._group_feature_slices(use_target=False)
        tgt_slices = self._group_feature_slices(use_target=True)
        for src_slice, tgt_slice in zip(src_slices, tgt_slices):
            W_block = fit_alignment(
                X[:, src_slice],
                Y[:, tgt_slice],
                method=base_method,
                lam=lam,
                rank=rank,
            )
            W[src_slice, tgt_slice] = W_block.to(dtype=X.dtype)
        return W

    @staticmethod
    def _transport_score(quality: dict[str, float]) -> float:
        return float(
            quality["mean_cosine_similarity"] - quality["relative_frobenius_error"]
        )

    def _group_signature(self, X: torch.Tensor) -> torch.Tensor:
        rank = max(1, int(self.config.transport_signature_rank))
        Xc = X.float() - X.float().mean(dim=0, keepdim=True)
        vals = torch.linalg.svdvals(Xc)
        vals = vals[: min(rank, vals.shape[0])]
        vals = vals / vals.sum().clamp_min(1e-8)
        if vals.shape[0] < rank:
            vals = torch.cat([vals, torch.zeros(rank - vals.shape[0], dtype=vals.dtype, device=vals.device)], dim=0)
        return vals

    def _canonical_subspace_rank(self, X: torch.Tensor, Y: torch.Tensor) -> int:
        rank_cfg = self.config.canonical_subspace_rank
        if rank_cfg is None:
            rank_cfg = self.config.transport_signature_rank
        return max(1, min(int(rank_cfg), X.shape[0], Y.shape[0], X.shape[1], Y.shape[1]))

    def _top_feature_basis(self, X: torch.Tensor, rank: int) -> torch.Tensor:
        Xc = X.float() - X.float().mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(Xc, full_matrices=False)
        basis = vh[:rank].T
        return basis.to(dtype=X.dtype)

    def _fit_preserve_projector(self, X: torch.Tensor, rank: int) -> torch.Tensor:
        rank = max(1, min(int(rank), X.shape[-1]))
        basis = self._top_feature_basis(X, rank).float()
        projector = basis @ basis.T
        projector = 0.5 * (projector + projector.T)
        return projector.to(dtype=X.dtype, device=X.device)

    def _fit_saliency_feature_weights(
        self,
        target: torch.Tensor,
        base_prediction: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        target_f = target.float()
        base_f = base_prediction.float()
        query_f = query_features.float()
        error_mag = (target_f - base_f).abs()
        query_mag = query_f.abs()
        signal = 0.5 * error_mag + 0.5 * target_f.abs()
        weights = (signal * (1.0 + query_mag)).mean(dim=0).clamp_min(1e-6)
        return (weights / weights.mean().clamp_min(1e-8)).to(dtype=target.dtype, device=target.device)

    def _fit_saliency_preserve_projector(
        self,
        target: torch.Tensor,
        base_prediction: torch.Tensor,
        query_features: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        weights = self._fit_saliency_feature_weights(target, base_prediction, query_features).float()
        rank = max(1, min(int(rank), weights.numel()))
        top_idx = torch.topk(weights, k=rank, largest=True).indices
        projector = torch.zeros(weights.numel(), weights.numel(), dtype=torch.float32, device=weights.device)
        projector[top_idx, top_idx] = 1.0
        return projector.to(dtype=target.dtype, device=target.device)

    def _fit_outlier_escrow_projector(
        self,
        target: torch.Tensor,
        base_prediction: torch.Tensor,
        query_features: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        target_f = target.float()
        base_f = base_prediction.float()
        query_f = query_features.float()
        error = (target_f - base_f).abs()
        score = (
            target_f.abs().amax(dim=0)
            + error.amax(dim=0)
            + target_f.std(dim=0, unbiased=False)
            + 0.25 * (target_f.abs() * (1.0 + query_f.abs())).mean(dim=0)
        )
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        rank = max(1, min(int(rank), score.numel()))
        top_idx = torch.topk(score, k=rank, largest=True).indices
        projector = torch.zeros(score.numel(), score.numel(), dtype=torch.float32, device=score.device)
        projector[top_idx, top_idx] = 1.0
        return projector.to(dtype=target.dtype, device=target.device)

    def _quantize_tail_with_preserve(
        self,
        values: torch.Tensor,
        preserve_proj: torch.Tensor,
    ) -> torch.Tensor:
        preserve = preserve_proj.to(device=values.device, dtype=values.dtype)
        complement = torch.eye(self.d_t, device=values.device, dtype=values.dtype) - preserve
        anchor = values @ preserve
        tail = values @ complement
        tail_q = self.quantizer.quantize_dequantize(tail)
        return anchor + tail_q

    def _fit_canonical_block_alignment(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        lam: float,
    ) -> torch.Tensor:
        rank = self._canonical_subspace_rank(X, Y)
        U_src = self._top_feature_basis(X, rank)
        U_tgt = self._top_feature_basis(Y, rank)
        Z_src = X @ U_src
        Z_tgt = Y @ U_tgt
        A = fit_alignment(Z_src, Z_tgt, method="procrustes", lam=lam)
        return U_src @ A @ U_tgt.T

    @staticmethod
    def _fix_basis_signs(basis: torch.Tensor) -> torch.Tensor:
        basis = basis.clone()
        if basis.numel() == 0:
            return basis
        max_idx = basis.abs().argmax(dim=0)
        signs = basis[max_idx, torch.arange(basis.shape[1], device=basis.device)]
        signs = torch.where(signs >= 0, torch.ones_like(signs), -torch.ones_like(signs))
        return basis * signs.unsqueeze(0)

    def _covariance_basis(
        self,
        X: torch.Tensor,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Xc = X.float() - X.float().mean(dim=0, keepdim=True)
        cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
        eps = 1e-4 * float(cov.trace().item() / max(1, cov.shape[0]) if cov.numel() > 0 else 1.0)
        cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        vals, vecs = torch.linalg.eigh(cov)
        vals = vals.flip(0)[:rank].clamp_min(1e-6)
        vecs = vecs.flip(1)[:, :rank]
        vecs = self._fix_basis_signs(vecs)
        return vecs.to(dtype=X.dtype), vals.to(dtype=X.dtype)

    def _fit_rotational_block_alignment(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        lam: float,
    ) -> torch.Tensor:
        rank = self._canonical_subspace_rank(X, Y)
        U_src, vals_src = self._covariance_basis(X, rank)
        U_tgt, vals_tgt = self._covariance_basis(Y, rank)
        scale_src = vals_src.rsqrt().to(dtype=X.dtype)
        scale_tgt = vals_tgt.sqrt().to(dtype=X.dtype)
        Z_src = (X @ U_src) * scale_src.unsqueeze(0)
        Z_tgt = (Y @ U_tgt) * vals_tgt.rsqrt().to(dtype=Y.dtype).unsqueeze(0)
        A = fit_alignment(Z_src, Z_tgt, method="procrustes", lam=lam)
        middle = scale_src.unsqueeze(1) * A.to(dtype=X.dtype)
        middle = middle * scale_tgt.unsqueeze(0)
        return U_src @ middle @ U_tgt.T

    def _fit_fitted_rotation_block_alignment(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        lam: float,
    ) -> torch.Tensor:
        del lam
        W_src, mean_src = fit_zca_whitening(X)
        W_tgt, mean_tgt = fit_zca_whitening(Y)
        X_white = apply_whitening(X, W_src, mean_src)
        Y_white = apply_whitening(Y, W_tgt, mean_tgt)
        cross = X_white.float().T @ Y_white.float()
        U, _, Vh = torch.linalg.svd(cross, full_matrices=False)
        rank = self._canonical_subspace_rank(X_white, Y_white)
        R = U[:, :rank] @ Vh[:rank, :]
        W_tgt_inv = torch.linalg.pinv(W_tgt.float()).to(dtype=Y.dtype)
        return W_src @ R.to(dtype=X.dtype) @ W_tgt_inv

    def _fit_shared_basis_block_alignment(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        lam: float,
    ) -> torch.Tensor:
        W_src, mean_src = fit_zca_whitening(X)
        W_tgt, mean_tgt = fit_zca_whitening(Y)
        X_white = apply_whitening(X, W_src, mean_src)
        Y_white = apply_whitening(Y, W_tgt, mean_tgt)
        cross = (X_white.float().T @ Y_white.float()) / max(1, X_white.shape[0] - 1)
        U, s, Vh = torch.linalg.svd(cross, full_matrices=False)
        rank = min(self._canonical_subspace_rank(X_white, Y_white), U.shape[1], Vh.shape[0], s.shape[0])
        U_r = U[:, :rank].to(dtype=X.dtype)
        V_r = Vh[:rank, :].T.to(dtype=Y.dtype)
        scales = s[:rank].clamp_min(1e-6).sqrt().to(dtype=X.dtype)
        Z_src = (X_white @ U_r) * scales.unsqueeze(0)
        Z_tgt = (Y_white @ V_r) * scales.to(dtype=Y.dtype).unsqueeze(0)
        A = fit_alignment(Z_src, Z_tgt, method="procrustes", lam=lam)
        W_tgt_inv = torch.linalg.pinv(W_tgt.float()).to(dtype=Y.dtype)
        return W_src @ U_r @ A.to(dtype=X.dtype) @ V_r.T @ W_tgt_inv

    def _fit_adaptive_canonical_block_alignment(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        lam: float,
        score_weight: float,
    ) -> torch.Tensor:
        candidates: list[torch.Tensor] = []
        method = "procrustes" if X.shape[1] == Y.shape[1] else "ridge"
        candidates.append(
            fit_alignment(
                X,
                Y,
                method=method,
                lam=lam,
            )
        )
        candidates.append(self._fit_canonical_block_alignment(X, Y, lam=lam))
        candidates.append(self._fit_rotational_block_alignment(X, Y, lam=lam))
        candidates.append(self._fit_fitted_rotation_block_alignment(X, Y, lam=lam))
        candidates.append(self._fit_shared_basis_block_alignment(X, Y, lam=lam))

        best_score = -float("inf")
        best_block = candidates[0]
        for W_block in candidates:
            q = alignment_quality(X, Y, W_block)
            score = self._transport_score(q)
            if score_weight > 0.0:
                y_hat = X @ W_block
                score = score - score_weight * float(self._subspace_distance(y_hat, Y))
            if score > best_score:
                best_score = score
                best_block = W_block
        return best_block

    def _subspace_distance(self, Y_hat: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        rank = max(1, int(self.config.transport_signature_rank))
        Yh = Y_hat.float() - Y_hat.float().mean(dim=0, keepdim=True)
        Yt = Y.float() - Y.float().mean(dim=0, keepdim=True)
        _, _, vh_hat = torch.linalg.svd(Yh, full_matrices=False)
        _, _, vh_tgt = torch.linalg.svd(Yt, full_matrices=False)
        r = min(rank, vh_hat.shape[0], vh_tgt.shape[0], Y.shape[1])
        basis_hat = vh_hat[:r].T
        basis_tgt = vh_tgt[:r].T
        proj_hat = basis_hat @ basis_hat.T
        proj_tgt = basis_tgt @ basis_tgt.T
        return (proj_hat - proj_tgt).pow(2).mean()

    def _covariance_distance(self, Y_hat: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        Yh = Y_hat.float() - Y_hat.float().mean(dim=0, keepdim=True)
        Yt = Y.float() - Y.float().mean(dim=0, keepdim=True)
        cov_hat = (Yh.T @ Yh) / max(1, Yh.shape[0] - 1)
        cov_tgt = (Yt.T @ Yt) / max(1, Yt.shape[0] - 1)
        cov_hat = cov_hat / cov_hat.trace().clamp_min(1e-8)
        cov_tgt = cov_tgt / cov_tgt.trace().clamp_min(1e-8)
        return (cov_hat - cov_tgt).pow(2).mean()

    def _template_distance(
        self,
        src_template: torch.Tensor,
        tgt_template: torch.Tensor,
    ) -> torch.Tensor:
        src = src_template.float()
        tgt = tgt_template.float()
        if src.numel() != tgt.numel():
            raise ValueError(
                "source and target group templates must agree on bin count, "
                f"got {src.numel()} and {tgt.numel()}"
            )
        src = src / src.sum().clamp_min(1e-8)
        tgt = tgt / tgt.sum().clamp_min(1e-8)
        mid = 0.5 * (src + tgt)
        return 0.5 * (
            (src * (src.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum()
            + (tgt * (tgt.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum()
        )

    def _descriptor_distance(
        self,
        src_descriptor: torch.Tensor,
        tgt_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        src = src_descriptor.float().view(-1)
        tgt = tgt_descriptor.float().view(-1)
        if src.numel() != tgt.numel():
            raise ValueError(
                "source and target descriptors must agree on size, "
                f"got {src.numel()} and {tgt.numel()}"
            )
        src = src / src.norm().clamp_min(1e-8)
        tgt = tgt / tgt.norm().clamp_min(1e-8)
        return 1.0 - torch.dot(src, tgt)

    def _contrastive_template_bonus(
        self,
        src_bank: torch.Tensor,
        tgt_bank: torch.Tensor,
    ) -> torch.Tensor:
        src = src_bank.float()
        tgt = tgt_bank.float()
        if src.ndim != 2 or tgt.ndim != 2:
            raise ValueError(
                "contrastive template banks must be 2D [num_prompts, bins], "
                f"got {tuple(src.shape)} and {tuple(tgt.shape)}"
            )
        if src.shape != tgt.shape:
            raise ValueError(
                "source and target template banks must agree on shape, "
                f"got {tuple(src.shape)} and {tuple(tgt.shape)}"
            )
        num_prompts = src.shape[0]
        if num_prompts == 0:
            raise ValueError("contrastive template banks must be non-empty")
        pos = torch.stack(
            [self._template_distance(src[prompt_idx], tgt[prompt_idx]) for prompt_idx in range(num_prompts)],
            dim=0,
        ).mean()
        if num_prompts == 1:
            return torch.zeros((), dtype=src.dtype, device=src.device)
        shifts = [1]
        half_shift = num_prompts // 2
        if num_prompts > 2 and half_shift not in {0, 1}:
            shifts.append(half_shift)
        neg_terms = []
        for shift in shifts:
            shifted = torch.roll(tgt, shifts=shift, dims=0)
            neg_terms.append(
                torch.stack(
                    [self._template_distance(src[prompt_idx], shifted[prompt_idx]) for prompt_idx in range(num_prompts)],
                    dim=0,
                ).mean()
            )
        neg = torch.stack(neg_terms, dim=0).mean()
        return neg - pos

    def _sinkhorn_transport(self, scores: torch.Tensor) -> torch.Tensor:
        temp = max(float(self.config.transport_temperature), 1e-6)
        log_plan = scores / temp
        log_plan = log_plan - log_plan.max()
        plan = torch.exp(log_plan)
        plan = plan.clamp_min(1e-8)
        for _ in range(max(1, int(self.config.transport_sinkhorn_iters))):
            plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
            plan = plan / plan.sum(dim=0, keepdim=True).clamp_min(1e-8)
        plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return plan

    def _rectangular_sinkhorn_transport(
        self,
        scores: torch.Tensor,
        *,
        row_mass: torch.Tensor,
        col_mass: torch.Tensor,
    ) -> torch.Tensor:
        temp = max(float(self.config.transport_temperature), 1e-6)
        kernel = torch.exp((scores - scores.max()) / temp).clamp_min(1e-8)
        u = torch.ones_like(row_mass)
        v = torch.ones_like(col_mass)
        for _ in range(max(1, int(self.config.transport_sinkhorn_iters))):
            u = row_mass / (kernel @ v).clamp_min(1e-8)
            v = col_mass / (kernel.T @ u).clamp_min(1e-8)
        plan = (u[:, None] * kernel) * v[None, :]
        return plan

    @staticmethod
    def _greedy_assignment(scores: torch.Tensor) -> torch.Tensor:
        n = scores.shape[0]
        plan = torch.zeros_like(scores)
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        flat = [
            (float(scores[i, j]), i, j)
            for i in range(n)
            for j in range(n)
        ]
        for _, i, j in sorted(flat, reverse=True):
            if i in used_rows or j in used_cols:
                continue
            plan[i, j] = 1.0
            used_rows.add(i)
            used_cols.add(j)
            if len(used_rows) == n:
                break
        return plan

    def _hard_transport_assignment(self, scores: torch.Tensor) -> torch.Tensor:
        n = scores.shape[0]
        if n <= 6:
            best_perm: tuple[int, ...] | None = None
            best_score = -float("inf")
            for perm in itertools.permutations(range(n)):
                score = sum(float(scores[i, perm[i]]) for i in range(n))
                if score > best_score:
                    best_score = score
                    best_perm = perm
            plan = torch.zeros_like(scores)
            assert best_perm is not None
            for i, j in enumerate(best_perm):
                plan[i, j] = 1.0
            return plan
        return self._greedy_assignment(scores)

    def _fit_group_transport_alignment(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        lam: float,
        protected_lam: float | None = None,
        protected_output_mask: torch.Tensor | None = None,
        protected_rank: int | None = None,
        residual_rank: int | None,
        src_layer_idx: int,
        tgt_layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_count, _, _ = self._head_group_layout()
        src_slices = self._group_feature_slices(use_target=False)
        tgt_slices = self._group_feature_slices(use_target=True)
        scores = torch.zeros(group_count, group_count, dtype=X.dtype, device=X.device)
        signature_weight = float(self.config.transport_signature_weight)
        src_signatures = None
        tgt_signatures = None
        src_templates = None
        tgt_templates = None
        if self.config.alignment_method == "grouped_signature_transport" and signature_weight > 0.0:
            src_signatures = [self._group_signature(X[:, src_slice]) for src_slice in src_slices]
            tgt_signatures = [self._group_signature(Y[:, tgt_slice]) for tgt_slice in tgt_slices]
        src_template_banks = None
        tgt_template_banks = None
        if self.config.alignment_method in {
            "grouped_template_transport",
            "grouped_qk_retrieval_transport",
            "grouped_template_subspace_transport",
        } and signature_weight > 0.0:
            if self._transport_src_group_templates is None or self._transport_tgt_group_templates is None:
                raise ValueError("grouped_template_transport requires calibration-time group templates")
            src_templates = self._transport_src_group_templates[src_layer_idx].to(device=X.device, dtype=X.dtype)
            tgt_templates = self._transport_tgt_group_templates[tgt_layer_idx].to(device=X.device, dtype=X.dtype)
            if src_templates.shape[0] != group_count or tgt_templates.shape[0] != group_count:
                raise ValueError(
                    "grouped template counts must match the transport group count, "
                    f"got {tuple(src_templates.shape)} and {tuple(tgt_templates.shape)} for {group_count} groups"
                )
        if self.config.alignment_method == "grouped_contrastive_template_transport" and signature_weight > 0.0:
            if self._transport_src_group_template_banks is None or self._transport_tgt_group_template_banks is None:
                raise ValueError("grouped_contrastive_template_transport requires calibration-time group template banks")
            src_template_banks = self._transport_src_group_template_banks[src_layer_idx].to(device=X.device, dtype=X.dtype)
            tgt_template_banks = self._transport_tgt_group_template_banks[tgt_layer_idx].to(device=X.device, dtype=X.dtype)
            if src_template_banks.ndim != 3 or tgt_template_banks.ndim != 3:
                raise ValueError(
                    "group template banks must have shape [num_prompts, group_count, bins], "
                    f"got {tuple(src_template_banks.shape)} and {tuple(tgt_template_banks.shape)}"
                )
            if src_template_banks.shape[1] != group_count or tgt_template_banks.shape[1] != group_count:
                raise ValueError(
                    "group template bank counts must match the transport group count, "
                    f"got {tuple(src_template_banks.shape)} and {tuple(tgt_template_banks.shape)} for {group_count} groups"
                )
        blocks: dict[tuple[int, int], torch.Tensor] = {}
        for src_idx, src_slice in enumerate(src_slices):
            for tgt_idx, tgt_slice in enumerate(tgt_slices):
                X_block = X[:, src_slice]
                Y_block = Y[:, tgt_slice]
                if self.config.alignment_method == "grouped_canonical_transport":
                    W_block = self._fit_canonical_block_alignment(
                        X_block,
                        Y_block,
                        lam=lam,
                    )
                elif self.config.alignment_method == "grouped_adaptive_canonical_transport":
                    W_block = self._fit_adaptive_canonical_block_alignment(
                        X_block,
                        Y_block,
                        lam=lam,
                        score_weight=signature_weight,
                    )
                elif self.config.alignment_method == "grouped_rotational_transport":
                    W_block = self._fit_rotational_block_alignment(
                        X_block,
                        Y_block,
                        lam=lam,
                    )
                elif self.config.alignment_method == "grouped_fitted_rotation_transport":
                    W_block = self._fit_fitted_rotation_block_alignment(
                        X_block,
                        Y_block,
                        lam=lam,
                    )
                elif self.config.alignment_method == "grouped_shared_basis_transport":
                    W_block = self._fit_shared_basis_block_alignment(
                        X_block,
                        Y_block,
                        lam=lam,
                    )
                else:
                    method = "procrustes" if (src_slice.stop - src_slice.start) == (tgt_slice.stop - tgt_slice.start) else "ridge"
                    W_block = fit_alignment(
                        X_block,
                        Y_block,
                        method=method,
                        lam=lam,
                    )
                q = alignment_quality(X[:, src_slice], Y[:, tgt_slice], W_block)
                score = self._transport_score(q)
                if self.config.alignment_method == "grouped_signature_transport" and signature_weight > 0.0:
                    sig_dist = (src_signatures[src_idx] - tgt_signatures[tgt_idx]).pow(2).mean()
                    score = score - signature_weight * float(sig_dist)
                elif self.config.alignment_method == "grouped_subspace_transport" and signature_weight > 0.0:
                    y_hat = X[:, src_slice] @ W_block
                    subspace_dist = self._subspace_distance(y_hat, Y[:, tgt_slice])
                    score = score - signature_weight * float(subspace_dist)
                elif self.config.alignment_method == "grouped_covariance_transport" and signature_weight > 0.0:
                    y_hat = X[:, src_slice] @ W_block
                    cov_dist = self._covariance_distance(y_hat, Y[:, tgt_slice])
                    score = score - signature_weight * float(cov_dist)
                elif self.config.alignment_method == "grouped_template_transport" and signature_weight > 0.0:
                    template_dist = self._template_distance(src_templates[src_idx], tgt_templates[tgt_idx])
                    score = score - signature_weight * float(template_dist)
                elif self.config.alignment_method == "grouped_qk_retrieval_transport" and signature_weight > 0.0:
                    template_dist = self._template_distance(src_templates[src_idx], tgt_templates[tgt_idx])
                    score = score - signature_weight * float(template_dist)
                elif self.config.alignment_method == "grouped_template_subspace_transport" and signature_weight > 0.0:
                    y_hat = X[:, src_slice] @ W_block
                    template_dist = self._template_distance(src_templates[src_idx], tgt_templates[tgt_idx])
                    subspace_dist = self._subspace_distance(y_hat, Y[:, tgt_slice])
                    score = score - signature_weight * float(template_dist + subspace_dist)
                elif self.config.alignment_method == "grouped_contrastive_template_transport" and signature_weight > 0.0:
                    contrast_bonus = self._contrastive_template_bonus(
                        src_template_banks[:, src_idx, :],
                        tgt_template_banks[:, tgt_idx, :],
                    )
                    score = score + signature_weight * float(contrast_bonus)
                scores[src_idx, tgt_idx] = score
                blocks[(src_idx, tgt_idx)] = W_block
        if self.config.alignment_method == "grouped_permutation":
            plan = self._hard_transport_assignment(scores)
        else:
            plan = self._sinkhorn_transport(scores)
        W = torch.zeros(self.d_s, self.d_t, dtype=X.dtype, device=X.device)
        for src_idx, src_slice in enumerate(src_slices):
            for tgt_idx, tgt_slice in enumerate(tgt_slices):
                W[src_slice, tgt_slice] = plan[src_idx, tgt_idx] * blocks[(src_idx, tgt_idx)].to(dtype=X.dtype)
        if residual_rank is not None and int(residual_rank) > 0:
            resid_rank = max(1, min(int(residual_rank), min(X.shape[1], Y.shape[1])))
            base_pred = X @ W
            residual_target = Y - base_pred
            effective_mask = protected_output_mask
            if effective_mask is None and protected_rank is not None:
                effective_mask = self._fit_ridge_top_output_mask(residual_target, int(protected_rank))
            W_resid = self._fit_alignment_with_protected_outputs(
                X,
                residual_target,
                method="reduced_rank",
                lam=lam,
                protected_lam=protected_lam,
                protected_output_mask=effective_mask,
                rank=resid_rank,
            )
            W = W + W_resid.to(dtype=X.dtype)
        return W, plan

    def _fit_broadcast_template_transport_alignment(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        lam: float,
        residual_rank: int | None,
        src_layer_idx: int,
        tgt_layer_idx: int,
        use_ot: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._transport_src_group_templates is None or self._transport_tgt_group_templates is None:
            raise ValueError("broadcast_template transport requires calibration-time head templates")
        src_slices = self._head_feature_slices(use_target=False)
        tgt_slices = self._head_feature_slices(use_target=True)
        src_templates = self._transport_src_group_templates[src_layer_idx].to(device=X.device, dtype=X.dtype)
        tgt_templates = self._transport_tgt_group_templates[tgt_layer_idx].to(device=X.device, dtype=X.dtype)
        if src_templates.shape[0] != len(src_slices) or tgt_templates.shape[0] != len(tgt_slices):
            raise ValueError(
                "broadcast template counts must match per-head counts, "
                f"got {tuple(src_templates.shape)} and {tuple(tgt_templates.shape)}"
            )
        scores = torch.zeros(len(src_slices), len(tgt_slices), dtype=X.dtype, device=X.device)
        blocks: dict[tuple[int, int], torch.Tensor] = {}
        weight = float(self.config.transport_signature_weight)
        for src_idx, src_slice in enumerate(src_slices):
            for tgt_idx, tgt_slice in enumerate(tgt_slices):
                W_block = fit_alignment(X[:, src_slice], Y[:, tgt_slice], method="ridge", lam=lam)
                q = alignment_quality(X[:, src_slice], Y[:, tgt_slice], W_block)
                score = self._transport_score(q)
                if weight > 0.0:
                    if self.config.alignment_method == "broadcast_qk_template_ot_transport":
                        template_dist = self._descriptor_distance(src_templates[src_idx], tgt_templates[tgt_idx])
                    else:
                        template_dist = self._template_distance(src_templates[src_idx], tgt_templates[tgt_idx])
                    score = score - weight * float(template_dist)
                scores[src_idx, tgt_idx] = score
                blocks[(src_idx, tgt_idx)] = W_block
        if use_ot:
            row_mass = torch.full(
                (len(src_slices),),
                float(len(tgt_slices)) / float(len(src_slices)),
                dtype=X.dtype,
                device=X.device,
            )
            col_mass = torch.ones(len(tgt_slices), dtype=X.dtype, device=X.device)
            plan = self._rectangular_sinkhorn_transport(scores, row_mass=row_mass, col_mass=col_mass)
        else:
            temp = max(float(self.config.transport_temperature), 1e-6)
            plan = torch.softmax(scores / temp, dim=1)
        W = torch.zeros(self.d_s, self.d_t, dtype=X.dtype, device=X.device)
        for src_idx, src_slice in enumerate(src_slices):
            for tgt_idx, tgt_slice in enumerate(tgt_slices):
                W[src_slice, tgt_slice] = plan[src_idx, tgt_idx] * blocks[(src_idx, tgt_idx)].to(dtype=X.dtype)
        if residual_rank is not None and int(residual_rank) > 0:
            resid_rank = max(1, min(int(residual_rank), min(X.shape[1], Y.shape[1])))
            base_pred = X @ W
            W_resid = fit_alignment(
                X,
                Y - base_pred,
                method="reduced_rank",
                lam=lam,
                rank=resid_rank,
            )
            W = W + W_resid.to(dtype=X.dtype)
        return W, plan

    def _target_head_group_ranges(self) -> list[tuple[int, int]]:
        group_count, _, tgt_heads_per_group = self._head_group_layout()
        return [
            (
                group_idx * tgt_heads_per_group,
                (group_idx + 1) * tgt_heads_per_group,
            )
            for group_idx in range(group_count)
        ]

    def _apply_head_selection_from_scores(self, group_scores: list[tuple[float, int]], tgt_layer_idx: int) -> None:
        group_count = len(group_scores)
        if self.config.head_selection_topk is not None:
            keep = max(1, min(group_count, self.config.head_selection_topk))
        else:
            keep = max(1, min(group_count, int(round(group_count * self.config.head_selection_ratio))))
        selected_groups = {group_idx for _, group_idx in sorted(group_scores, reverse=True)[:keep]}
        mask = torch.zeros(self.config.tgt_num_heads, dtype=torch.bool, device=self.head_selected_mask.device)
        for group_idx, (start, end) in enumerate(self._target_head_group_ranges()):
            if group_idx in selected_groups:
                mask[start:end] = True
        self.head_selected_mask[tgt_layer_idx].copy_(mask)

    def _fit_pre_quant_filter(self, Y: torch.Tensor) -> torch.Tensor:
        rank = self.config.pre_quant_rank
        shrinkage = float(self.config.pre_quant_shrinkage)
        if rank is not None and int(rank) <= 0:
            return torch.eye(self.d_t, dtype=Y.dtype, device=Y.device)
        if rank is None and shrinkage <= 0.0:
            return torch.eye(self.d_t, dtype=Y.dtype, device=Y.device)
        Yc = Y.float() - Y.float().mean(dim=0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(Yc, full_matrices=False)
        if rank is None:
            rank = vh.shape[0]
        rank = max(1, min(int(rank), vh.shape[0], self.d_t))
        basis = vh[:rank].T.to(dtype=Y.dtype, device=Y.device)
        kept = singular_values[:rank].to(dtype=Y.dtype, device=Y.device)
        if shrinkage <= 0.0:
            weights = torch.ones_like(kept)
        else:
            tau = shrinkage * kept.mean().clamp_min(1e-8)
            weights = (kept / (kept + tau)).clamp(0.0, 1.0)
        return basis @ torch.diag(weights) @ basis.T

    def _fit_affine_correction(self, quantized: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = quantized.float()
        y = target.float()
        q_mean = q.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        q_center = q - q_mean
        y_center = y - y_mean
        var_q = q_center.pow(2).mean(dim=0, keepdim=True).clamp_min(1e-8)
        cov_qy = (q_center * y_center).mean(dim=0, keepdim=True)
        scale = cov_qy / var_q
        bias = y_mean - scale * q_mean
        return scale.to(dtype=target.dtype, device=target.device), bias.to(dtype=target.dtype, device=target.device)

    def _fit_bridge_affine_correction(
        self,
        quantized: torch.Tensor,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = quantized.float()
        p = predicted.float()
        y = target.float()
        lam = float(self.config.ridge_lambda)

        a11 = (q * q).sum(dim=0) + lam
        a22 = (p * p).sum(dim=0) + lam
        a12 = (q * p).sum(dim=0)
        a13 = q.sum(dim=0)
        a23 = p.sum(dim=0)
        a33 = torch.full_like(a11, float(q.shape[0]))
        b1 = (q * y).sum(dim=0)
        b2 = (p * y).sum(dim=0)
        b3 = y.sum(dim=0)

        mat = torch.stack(
            [
                torch.stack([a11, a12, a13], dim=-1),
                torch.stack([a12, a22, a23], dim=-1),
                torch.stack([a13, a23, a33], dim=-1),
            ],
            dim=-2,
        )
        rhs = torch.stack([b1, b2, b3], dim=-1).unsqueeze(-1)
        try:
            sol = torch.linalg.solve(mat, rhs).squeeze(-1)
        except RuntimeError:
            eye = torch.eye(3, dtype=mat.dtype, device=mat.device).unsqueeze(0)
            sol = torch.linalg.solve(mat + 1e-4 * eye, rhs).squeeze(-1)
        q_scale = sol[:, 0].unsqueeze(0)
        aux_scale = sol[:, 1].unsqueeze(0)
        bias = sol[:, 2].unsqueeze(0)
        return (
            q_scale.to(dtype=target.dtype, device=target.device),
            aux_scale.to(dtype=target.dtype, device=target.device),
            bias.to(dtype=target.dtype, device=target.device),
        )

    def _fit_bridge_ridge_correction(
        self,
        quantized: torch.Tensor,
        predicted: torch.Tensor,
        target: torch.Tensor,
        *,
        lam: float,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([quantized.float(), predicted.float()], dim=-1)
        y = target.float()
        if sample_weights is not None:
            weights = sample_weights.float().view(-1, 1)
            if weights.shape[0] != x.shape[0]:
                raise ValueError(
                    "sample_weights must align with bridge samples, "
                    f"got {weights.shape[0]} vs {x.shape[0]}"
                )
            weights = weights / weights.mean().clamp_min(1e-8)
            norm = weights.sum().clamp_min(1e-8)
            x_mean = (weights * x).sum(dim=0, keepdim=True) / norm
            y_mean = (weights * y).sum(dim=0, keepdim=True) / norm
        else:
            weights = None
            x_mean = x.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
        x_center = x - x_mean
        y_center = y - y_mean
        if weights is not None:
            sqrt_w = weights.sqrt()
            x_weighted = x_center * sqrt_w
            y_weighted = y_center * sqrt_w
            xtx = x_weighted.T @ x_weighted
            rhs = x_weighted.T @ y_weighted
        else:
            xtx = x_center.T @ x_center
            rhs = x_center.T @ y_center
        eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
        system = xtx + lam * eye
        try:
            weight = torch.linalg.solve(system, rhs)
        except RuntimeError:
            weight = torch.linalg.pinv(system) @ rhs
        bias = y_mean - x_mean @ weight
        q_weight, aux_weight = weight[: self.d_t], weight[self.d_t :]
        return (
            q_weight.to(dtype=target.dtype, device=target.device),
            aux_weight.to(dtype=target.dtype, device=target.device),
            bias.to(dtype=target.dtype, device=target.device),
        )

    def _fit_bridge_ridge_query_projector_correction(
        self,
        quantized: torch.Tensor,
        predicted: torch.Tensor,
        query_features: torch.Tensor,
        target: torch.Tensor,
        *,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qfeat = query_features.float()
        if qfeat.shape != quantized.shape:
            raise ValueError(
                "query_features must match quantized/predicted bridge samples, "
                f"got {tuple(qfeat.shape)} vs {tuple(quantized.shape)}"
            )
        x = torch.cat(
            [
                quantized.float(),
                predicted.float(),
                quantized.float() * qfeat,
                predicted.float() * qfeat,
            ],
            dim=-1,
        )
        y = target.float()
        x_mean = x.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        x_center = x - x_mean
        y_center = y - y_mean
        xtx = x_center.T @ x_center
        eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
        rhs = x_center.T @ y_center
        system = xtx + lam * eye
        try:
            weight = torch.linalg.solve(system, rhs)
        except RuntimeError:
            weight = torch.linalg.pinv(system) @ rhs
        bias = y_mean - x_mean @ weight
        q_weight = weight[: self.d_t]
        aux_weight = weight[self.d_t : 2 * self.d_t]
        q_query_weight = weight[2 * self.d_t : 3 * self.d_t]
        aux_query_weight = weight[3 * self.d_t :]
        return (
            q_weight.to(dtype=target.dtype, device=target.device),
            aux_weight.to(dtype=target.dtype, device=target.device),
            q_query_weight.to(dtype=target.dtype, device=target.device),
            aux_query_weight.to(dtype=target.dtype, device=target.device),
            bias.to(dtype=target.dtype, device=target.device),
        )

    def _fit_bridge_query_residual_adapter(
        self,
        quantized: torch.Tensor,
        predicted: torch.Tensor,
        query_features: torch.Tensor,
        base_prediction: torch.Tensor,
        residual_target: torch.Tensor,
        *,
        rank: int,
        steps: int = 80,
        lr: float = 5e-2,
        logit_weight: float = 0.5,
        affinity_weight: float = 0.0,
        affinity_batch_size: int = 256,
        attention_kl_weight: float = 0.0,
        attention_kl_batch_size: int = 128,
        local_attention_weight: float = 0.0,
        interaction_distill_weight: float = 0.0,
        readout_distill_weight: float = 0.0,
        prediction_distill_weight: float = 0.0,
        teacher_topk_log_probs: torch.Tensor | None = None,
        teacher_topk_output_rows: torch.Tensor | None = None,
        readout_partner: torch.Tensor | None = None,
        readout_partner_kind: str | None = None,
        sample_prompt_ids: torch.Tensor | None = None,
        local_attention_prompt_batch_size: int = 4,
        interaction_prompt_batch_size: int = 4,
        interaction_temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = quantized.float()
        p = predicted.float()
        query = query_features.float()
        base = base_prediction.float()
        target = residual_target.float()
        if q.shape != p.shape or q.shape != query.shape or q.shape != base.shape or q.shape != target.shape:
            raise ValueError(
                "quantized, predicted, query_features, base_prediction, and residual_target must all match, "
                f"got {tuple(q.shape)}, {tuple(p.shape)}, {tuple(query.shape)}, {tuple(base.shape)}, and {tuple(target.shape)}"
            )
        rank = max(1, min(int(rank), self.d_t))
        gen = torch.Generator(device="cpu").manual_seed(223_901 + int(self.config.seed) * 97 + int(rank))
        scale = 1e-2
        q_left = torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale
        q_right = torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale
        aux_left = torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale
        aux_right = torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale
        q_left = torch.nn.Parameter(q_left)
        q_right = torch.nn.Parameter(q_right)
        aux_left = torch.nn.Parameter(aux_left)
        aux_right = torch.nn.Parameter(aux_right)
        optimizer = torch.optim.Adam([q_left, q_right, aux_left, aux_right], lr=float(lr))
        prompt_ids_cpu = None
        unique_prompt_ids = None
        if (
            float(local_attention_weight) > 0.0
            or float(interaction_distill_weight) > 0.0
            or float(readout_distill_weight) > 0.0
        ):
            if sample_prompt_ids is None:
                raise ValueError(
                    "sample_prompt_ids are required when local_attention_weight > 0, "
                    "interaction_distill_weight > 0, or readout_distill_weight > 0"
                )
            prompt_ids_cpu = sample_prompt_ids.detach().to("cpu", dtype=torch.long).view(-1)
            if prompt_ids_cpu.numel() != q.shape[0]:
                raise ValueError(
                    "sample_prompt_ids must align with calibration samples, "
                    f"got {int(prompt_ids_cpu.numel())} ids for {q.shape[0]} samples"
                )
            unique_prompt_ids = torch.unique(prompt_ids_cpu)
        partner = None
        if float(readout_distill_weight) > 0.0:
            if readout_partner is None:
                raise ValueError("readout_partner is required when readout_distill_weight > 0")
            if readout_partner_kind not in {"K", "V"}:
                raise ValueError("readout_partner_kind must be 'K' or 'V' when readout_distill_weight > 0")
            partner = readout_partner.float()
            if partner.shape != target.shape:
                raise ValueError(
                    "readout_partner must match residual_target shape, "
                    f"got {tuple(partner.shape)} vs {tuple(target.shape)}"
                )
        teacher_log_probs = None
        teacher_output_rows = None
        if float(prediction_distill_weight) > 0.0:
            if teacher_topk_log_probs is None or teacher_topk_output_rows is None:
                raise ValueError(
                    "teacher_topk_log_probs and teacher_topk_output_rows are required when prediction_distill_weight > 0"
                )
            teacher_log_probs = teacher_topk_log_probs.float()
            teacher_output_rows = teacher_topk_output_rows.float()
            if teacher_log_probs.ndim != 2 or teacher_log_probs.shape[0] != q.shape[0]:
                raise ValueError(
                    "teacher_topk_log_probs must be [samples, topk] aligned with calibration samples, "
                    f"got {tuple(teacher_log_probs.shape)} for {q.shape[0]} samples"
                )
            if teacher_output_rows.ndim != 3 or teacher_output_rows.shape[:2] != teacher_log_probs.shape:
                raise ValueError(
                    "teacher_topk_output_rows must be [samples, topk, d_t] aligned with teacher_topk_log_probs, "
                    f"got {tuple(teacher_output_rows.shape)} vs {tuple(teacher_log_probs.shape)}"
                )
            if teacher_output_rows.shape[-1] != self.d_t:
                raise ValueError(
                    f"teacher_topk_output_rows width {teacher_output_rows.shape[-1]} does not match target width {self.d_t}"
                )

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                pred = (q * query) @ q_left @ q_right + (p * query) @ aux_left @ aux_right
                recon_loss = F.mse_loss(pred, target)
                logit_pred = (pred * query).sum(dim=-1)
                logit_tgt = (target * query).sum(dim=-1)
                loss = recon_loss + float(logit_weight) * F.mse_loss(logit_pred, logit_tgt)
                if float(affinity_weight) > 0.0:
                    if q.shape[0] > int(affinity_batch_size):
                        affinity_idx = torch.randperm(q.shape[0], generator=gen)[: int(affinity_batch_size)]
                    else:
                        affinity_idx = slice(None)
                    full_pred = base[affinity_idx] + pred[affinity_idx]
                    full_tgt = base[affinity_idx] + target[affinity_idx]
                    affinity_pred = F.normalize(full_pred * query[affinity_idx], dim=-1)
                    affinity_tgt = F.normalize(full_tgt * query[affinity_idx], dim=-1)
                    aff_pred = affinity_pred @ affinity_pred.T
                    aff_tgt = affinity_tgt @ affinity_tgt.T
                    loss = loss + float(affinity_weight) * F.mse_loss(aff_pred, aff_tgt)
                if float(attention_kl_weight) > 0.0:
                    if q.shape[0] > int(attention_kl_batch_size):
                        attn_idx = torch.randperm(q.shape[0], generator=gen)[: int(attention_kl_batch_size)]
                    else:
                        attn_idx = slice(None)
                    q_batch = query[attn_idx]
                    full_pred = base[attn_idx] + pred[attn_idx]
                    full_tgt = base[attn_idx] + target[attn_idx]
                    scale = math.sqrt(float(self.d_t))
                    logits_pred = (q_batch @ full_pred.T) / scale
                    logits_tgt = (q_batch @ full_tgt.T) / scale
                    tgt_probs = torch.softmax(logits_tgt, dim=-1)
                    pred_log_probs = torch.log_softmax(logits_pred, dim=-1)
                    loss = loss + float(attention_kl_weight) * F.kl_div(pred_log_probs, tgt_probs, reduction="batchmean")
                if float(local_attention_weight) > 0.0 and unique_prompt_ids is not None and prompt_ids_cpu is not None:
                    if unique_prompt_ids.numel() > int(local_attention_prompt_batch_size):
                        prompt_choice = unique_prompt_ids[
                            torch.randperm(unique_prompt_ids.numel(), generator=gen)[: int(local_attention_prompt_batch_size)]
                        ]
                    else:
                        prompt_choice = unique_prompt_ids
                    local_losses: list[torch.Tensor] = []
                    for prompt_id in prompt_choice.tolist():
                        prompt_idx = torch.nonzero(prompt_ids_cpu == int(prompt_id), as_tuple=False).view(-1)
                        if prompt_idx.numel() <= 1:
                            continue
                        prompt_idx = prompt_idx.to(device=q.device)
                        q_prompt = query[prompt_idx]
                        full_pred = base[prompt_idx] + pred[prompt_idx]
                        full_tgt = base[prompt_idx] + target[prompt_idx]
                        scale = math.sqrt(float(self.d_t))
                        logits_pred = (q_prompt @ full_pred.T) / scale
                        logits_tgt = (q_prompt @ full_tgt.T) / scale
                        causal_mask = torch.triu(
                            torch.ones(logits_pred.shape[0], logits_pred.shape[1], device=logits_pred.device, dtype=torch.bool),
                            diagonal=1,
                        )
                        logits_pred = logits_pred.masked_fill(causal_mask, -1e9)
                        logits_tgt = logits_tgt.masked_fill(causal_mask, -1e9)
                        tgt_probs = torch.softmax(logits_tgt, dim=-1)
                        pred_log_probs = torch.log_softmax(logits_pred, dim=-1)
                        local_losses.append(F.kl_div(pred_log_probs, tgt_probs, reduction="batchmean"))
                    if local_losses:
                        loss = loss + float(local_attention_weight) * torch.stack(local_losses).mean()
                if float(interaction_distill_weight) > 0.0 and unique_prompt_ids is not None and prompt_ids_cpu is not None:
                    if unique_prompt_ids.numel() > int(interaction_prompt_batch_size):
                        prompt_choice = unique_prompt_ids[
                            torch.randperm(unique_prompt_ids.numel(), generator=gen)[: int(interaction_prompt_batch_size)]
                        ]
                    else:
                        prompt_choice = unique_prompt_ids
                    interaction_losses: list[torch.Tensor] = []
                    temperature = max(float(interaction_temperature), 1e-4)
                    for prompt_id in prompt_choice.tolist():
                        prompt_idx = torch.nonzero(prompt_ids_cpu == int(prompt_id), as_tuple=False).view(-1)
                        if prompt_idx.numel() <= 1:
                            continue
                        prompt_idx = prompt_idx.to(device=q.device)
                        full_pred = base[prompt_idx] + pred[prompt_idx]
                        full_tgt = base[prompt_idx] + target[prompt_idx]
                        feat_pred = F.normalize(full_pred, dim=-1)
                        feat_tgt = F.normalize(full_tgt, dim=-1)
                        sim_pred = (feat_pred @ feat_pred.T) / temperature
                        sim_tgt = (feat_tgt @ feat_tgt.T) / temperature
                        eye_mask = torch.eye(sim_pred.shape[0], device=sim_pred.device, dtype=torch.bool)
                        sim_pred = sim_pred.masked_fill(eye_mask, -1e9)
                        sim_tgt = sim_tgt.masked_fill(eye_mask, -1e9)
                        tgt_probs = torch.softmax(sim_tgt, dim=-1)
                        pred_log_probs = torch.log_softmax(sim_pred, dim=-1)
                        interaction_losses.append(F.kl_div(pred_log_probs, tgt_probs, reduction="batchmean"))
                    if interaction_losses:
                        loss = loss + float(interaction_distill_weight) * torch.stack(interaction_losses).mean()
                if (
                    float(readout_distill_weight) > 0.0
                    and unique_prompt_ids is not None
                    and prompt_ids_cpu is not None
                    and partner is not None
                ):
                    if unique_prompt_ids.numel() > int(local_attention_prompt_batch_size):
                        prompt_choice = unique_prompt_ids[
                            torch.randperm(unique_prompt_ids.numel(), generator=gen)[: int(local_attention_prompt_batch_size)]
                        ]
                    else:
                        prompt_choice = unique_prompt_ids
                    readout_losses: list[torch.Tensor] = []
                    scale = math.sqrt(float(self.config.tgt_head_dim))
                    for prompt_id in prompt_choice.tolist():
                        prompt_idx = torch.nonzero(prompt_ids_cpu == int(prompt_id), as_tuple=False).view(-1)
                        if prompt_idx.numel() <= 1:
                            continue
                        prompt_idx = prompt_idx.to(device=q.device)
                        q_prompt = query[prompt_idx].view(-1, self.config.tgt_num_heads, self.config.tgt_head_dim)
                        full_pred = (base[prompt_idx] + pred[prompt_idx]).view(
                            -1, self.config.tgt_num_heads, self.config.tgt_head_dim
                        )
                        full_tgt = (base[prompt_idx] + target[prompt_idx]).view(
                            -1, self.config.tgt_num_heads, self.config.tgt_head_dim
                        )
                        partner_prompt = partner[prompt_idx].view(
                            -1, self.config.tgt_num_heads, self.config.tgt_head_dim
                        )
                        if readout_partner_kind == "V":
                            logits_pred = torch.einsum("thd,shd->ths", q_prompt, full_pred) / scale
                            logits_tgt = torch.einsum("thd,shd->ths", q_prompt, full_tgt) / scale
                            causal_mask = torch.triu(
                                torch.ones(logits_pred.shape[0], logits_pred.shape[2], device=logits_pred.device, dtype=torch.bool),
                                diagonal=1,
                            ).unsqueeze(1)
                            logits_pred = logits_pred.masked_fill(causal_mask, -1e9)
                            logits_tgt = logits_tgt.masked_fill(causal_mask, -1e9)
                            weights_pred = torch.softmax(logits_pred, dim=-1)
                            weights_tgt = torch.softmax(logits_tgt, dim=-1)
                            readout_pred = torch.einsum("ths,shd->thd", weights_pred, partner_prompt)
                            readout_tgt = torch.einsum("ths,shd->thd", weights_tgt, partner_prompt)
                        else:
                            logits = torch.einsum("thd,shd->ths", q_prompt, partner_prompt) / scale
                            causal_mask = torch.triu(
                                torch.ones(logits.shape[0], logits.shape[2], device=logits.device, dtype=torch.bool),
                                diagonal=1,
                            ).unsqueeze(1)
                            logits = logits.masked_fill(causal_mask, -1e9)
                            weights = torch.softmax(logits, dim=-1)
                            readout_pred = torch.einsum("ths,shd->thd", weights, full_pred)
                            readout_tgt = torch.einsum("ths,shd->thd", weights, full_tgt)
                        readout_losses.append(F.mse_loss(readout_pred, readout_tgt))
                    if readout_losses:
                        loss = loss + float(readout_distill_weight) * torch.stack(readout_losses).mean()
                if (
                    float(prediction_distill_weight) > 0.0
                    and teacher_log_probs is not None
                    and teacher_output_rows is not None
                ):
                    teacher_probs = torch.softmax(
                        teacher_log_probs.to(device=q.device, dtype=torch.float32),
                        dim=-1,
                    )
                    teacher_rows = teacher_output_rows.to(device=q.device, dtype=torch.float32)
                    hidden_pred = (base + pred) * query
                    student_logits = torch.einsum("nd,nkd->nk", hidden_pred, teacher_rows) / math.sqrt(float(self.d_t))
                    student_log_probs = torch.log_softmax(student_logits, dim=-1)
                    loss = loss + float(prediction_distill_weight) * F.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction="batchmean",
                    )
                loss.backward()
                optimizer.step()

        return (
            q_left.detach().to(dtype=residual_target.dtype, device=residual_target.device),
            q_right.detach().to(dtype=residual_target.dtype, device=residual_target.device),
            aux_left.detach().to(dtype=residual_target.dtype, device=residual_target.device),
            aux_right.detach().to(dtype=residual_target.dtype, device=residual_target.device),
        )

    def _fit_bridge_query_shared_residual_adapter(
        self,
        quantized_k: torch.Tensor,
        predicted_k: torch.Tensor,
        quantized_v: torch.Tensor,
        predicted_v: torch.Tensor,
        query_features: torch.Tensor,
        base_prediction_k: torch.Tensor,
        base_prediction_v: torch.Tensor,
        residual_target_k: torch.Tensor,
        residual_target_v: torch.Tensor,
        *,
        rank: int,
        steps: int = 100,
        lr: float = 5e-2,
        logit_weight: float = 0.5,
        prediction_distill_weight: float = 0.0,
        dynamic_prediction_weight: float = 0.0,
        teacher_topk_log_probs: torch.Tensor | None = None,
        teacher_topk_output_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        tensors = (
            quantized_k.float(),
            predicted_k.float(),
            quantized_v.float(),
            predicted_v.float(),
            query_features.float(),
            base_prediction_k.float(),
            base_prediction_v.float(),
            residual_target_k.float(),
            residual_target_v.float(),
        )
        shapes = sorted({tuple(t.shape) for t in tensors})
        if len(shapes) != 1:
            raise ValueError(f"all shared adapter tensors must match, got shapes {shapes}")

        qk, pk, qv, pv, query, base_k, base_v, target_k, target_v = tensors
        rank = max(1, min(int(rank), self.d_t))
        gen = torch.Generator(device="cpu").manual_seed(871_337 + int(self.config.seed) * 131 + int(rank))
        scale = 1e-2

        shared_left = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        shared_aux_left = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        shared_k_right = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        shared_v_right = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        priv_k_left = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        priv_k_right = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        priv_k_aux_left = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        priv_k_aux_right = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        priv_v_left = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        priv_v_right = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        priv_v_aux_left = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        priv_v_aux_right = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)

        optimizer = torch.optim.Adam(
            [
                shared_left,
                shared_aux_left,
                shared_k_right,
                shared_v_right,
                priv_k_left,
                priv_k_right,
                priv_k_aux_left,
                priv_k_aux_right,
                priv_v_left,
                priv_v_right,
                priv_v_aux_left,
                priv_v_aux_right,
            ],
            lr=float(lr),
        )

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                shared_query = 0.5 * ((qk + qv) * query)
                shared_aux = 0.5 * ((pk + pv) * query)
                shared_latent = shared_query @ shared_left + shared_aux @ shared_aux_left

                pred_k = (
                    shared_latent @ shared_k_right
                    + ((qk * query) @ priv_k_left) @ priv_k_right
                    + ((pk * query) @ priv_k_aux_left) @ priv_k_aux_right
                )
                pred_v = (
                    shared_latent @ shared_v_right
                    + ((qv * query) @ priv_v_left) @ priv_v_right
                    + ((pv * query) @ priv_v_aux_left) @ priv_v_aux_right
                )

                loss = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)
                logit_pred_k = ((base_k + pred_k) * query).sum(dim=-1)
                logit_tgt_k = ((base_k + target_k) * query).sum(dim=-1)
                logit_pred_v = ((base_v + pred_v) * query).sum(dim=-1)
                logit_tgt_v = ((base_v + target_v) * query).sum(dim=-1)
                loss = loss + float(logit_weight) * (
                    F.mse_loss(logit_pred_k, logit_tgt_k) + F.mse_loss(logit_pred_v, logit_tgt_v)
                )
                if (
                    (float(prediction_distill_weight) > 0.0 or float(dynamic_prediction_weight) > 0.0)
                    and teacher_topk_log_probs is not None
                    and teacher_topk_output_rows is not None
                ):
                    teacher_logits = teacher_topk_log_probs.to(device=qk.device, dtype=torch.float32)
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)
                    teacher_rows = teacher_topk_output_rows.to(device=qk.device, dtype=torch.float32)
                    hidden_pred = 0.5 * (((base_k + pred_k) * query) + ((base_v + pred_v) * query))
                    student_logits = torch.einsum("nd,nkd->nk", hidden_pred, teacher_rows) / math.sqrt(float(self.d_t))
                    student_log_probs = torch.log_softmax(student_logits, dim=-1)
                    if float(prediction_distill_weight) > 0.0:
                        loss = loss + float(prediction_distill_weight) * F.kl_div(
                            student_log_probs,
                            teacher_probs,
                            reduction="batchmean",
                        )
                    if float(dynamic_prediction_weight) > 0.0:
                        context_hidden = 0.5 * ((qk + qv) * query)
                        context_logits = torch.einsum("nd,nkd->nk", context_hidden, teacher_rows) / math.sqrt(float(self.d_t))
                        dynamic_teacher_probs = torch.softmax(teacher_logits + context_logits, dim=-1)
                        loss = loss + float(dynamic_prediction_weight) * F.kl_div(
                            student_log_probs,
                            dynamic_teacher_probs,
                            reduction="batchmean",
                        )
                loss.backward()
                optimizer.step()

        return (
            shared_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            shared_aux_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            shared_k_right.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            shared_v_right.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
            priv_k_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            priv_k_right.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            priv_k_aux_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            priv_k_aux_right.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            priv_v_left.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
            priv_v_right.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
            priv_v_aux_left.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
            priv_v_aux_right.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
        )

    def _fit_bridge_query_xattn_adapter(
        self,
        quantized_k: torch.Tensor,
        predicted_k: torch.Tensor,
        quantized_v: torch.Tensor,
        predicted_v: torch.Tensor,
        query_features: torch.Tensor,
        base_prediction_k: torch.Tensor,
        base_prediction_v: torch.Tensor,
        residual_target_k: torch.Tensor,
        residual_target_v: torch.Tensor,
        *,
        rank: int,
        steps: int = 100,
        lr: float = 5e-2,
        logit_weight: float = 0.5,
        prediction_distill_weight: float = 0.0,
        dynamic_prediction_weight: float = 0.0,
        teacher_topk_log_probs: torch.Tensor | None = None,
        teacher_topk_output_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        tensors = (
            quantized_k.float(),
            predicted_k.float(),
            quantized_v.float(),
            predicted_v.float(),
            query_features.float(),
            base_prediction_k.float(),
            base_prediction_v.float(),
            residual_target_k.float(),
            residual_target_v.float(),
        )
        shapes = sorted({tuple(t.shape) for t in tensors})
        if len(shapes) != 1:
            raise ValueError(f"all cross-attention adapter tensors must match, got shapes {shapes}")

        qk, pk, qv, pv, query, base_k, base_v, target_k, target_v = tensors
        rank = max(1, min(int(rank), self.d_t))
        gen = torch.Generator(device="cpu").manual_seed(917_411 + int(self.config.seed) * 149 + int(rank))
        scale = 1e-2

        q_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        k_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        v_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        out_k = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        out_v = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        optimizer = torch.optim.Adam([q_proj, k_proj, v_proj, out_k, out_v], lr=float(lr))

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                memory = torch.stack([qk, pk, qv, pv], dim=1)
                q_hidden = query @ q_proj
                key_hidden = torch.einsum("nmd,dr->nmr", memory, k_proj)
                value_hidden = torch.einsum("nmd,dr->nmr", memory, v_proj)
                attn_logits = torch.einsum("nr,nmr->nm", q_hidden, key_hidden) / math.sqrt(float(rank))
                attn = torch.softmax(attn_logits, dim=-1)
                context = torch.einsum("nm,nmr->nr", attn, value_hidden)
                pred_k = context @ out_k
                pred_v = context @ out_v
                loss = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)
                logit_pred_k = ((base_k + pred_k) * query).sum(dim=-1)
                logit_tgt_k = ((base_k + target_k) * query).sum(dim=-1)
                logit_pred_v = ((base_v + pred_v) * query).sum(dim=-1)
                logit_tgt_v = ((base_v + target_v) * query).sum(dim=-1)
                loss = loss + float(logit_weight) * (
                    F.mse_loss(logit_pred_k, logit_tgt_k) + F.mse_loss(logit_pred_v, logit_tgt_v)
                )
                if (
                    (float(prediction_distill_weight) > 0.0 or float(dynamic_prediction_weight) > 0.0)
                    and teacher_topk_log_probs is not None
                    and teacher_topk_output_rows is not None
                ):
                    teacher_logits = teacher_topk_log_probs.to(device=qk.device, dtype=torch.float32)
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)
                    teacher_rows = teacher_topk_output_rows.to(device=qk.device, dtype=torch.float32)
                    hidden_pred = 0.5 * (((base_k + pred_k) * query) + ((base_v + pred_v) * query))
                    student_logits = torch.einsum("nd,nkd->nk", hidden_pred, teacher_rows) / math.sqrt(float(self.d_t))
                    student_log_probs = torch.log_softmax(student_logits, dim=-1)
                    if float(prediction_distill_weight) > 0.0:
                        loss = loss + float(prediction_distill_weight) * F.kl_div(
                            student_log_probs,
                            teacher_probs,
                            reduction="batchmean",
                        )
                    if float(dynamic_prediction_weight) > 0.0:
                        context_hidden = 0.5 * ((qk + qv) * query)
                        context_logits = torch.einsum("nd,nkd->nk", context_hidden, teacher_rows) / math.sqrt(float(self.d_t))
                        dynamic_teacher_probs = torch.softmax(teacher_logits + context_logits, dim=-1)
                        loss = loss + float(dynamic_prediction_weight) * F.kl_div(
                            student_log_probs,
                            dynamic_teacher_probs,
                            reduction="batchmean",
                        )
                loss.backward()
                optimizer.step()

        return (
            q_proj.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            k_proj.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            v_proj.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            out_k.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            out_v.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
        )

    def _fit_bridge_query_module_adapter(
        self,
        quantized_k: torch.Tensor,
        predicted_k: torch.Tensor,
        quantized_v: torch.Tensor,
        predicted_v: torch.Tensor,
        query_features: torch.Tensor,
        base_prediction_k: torch.Tensor,
        base_prediction_v: torch.Tensor,
        residual_target_k: torch.Tensor,
        residual_target_v: torch.Tensor,
        *,
        rank: int,
        steps: int = 100,
        lr: float = 5e-2,
        logit_weight: float = 0.5,
        prediction_distill_weight: float = 0.0,
        dynamic_prediction_weight: float = 0.0,
        teacher_topk_log_probs: torch.Tensor | None = None,
        teacher_topk_output_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        tensors = (
            quantized_k.float(),
            predicted_k.float(),
            quantized_v.float(),
            predicted_v.float(),
            query_features.float(),
            base_prediction_k.float(),
            base_prediction_v.float(),
            residual_target_k.float(),
            residual_target_v.float(),
        )
        shapes = sorted({tuple(t.shape) for t in tensors})
        if len(shapes) != 1:
            raise ValueError(f"all module-adapter tensors must match, got shapes {shapes}")

        qk, pk, qv, pv, query, base_k, base_v, target_k, target_v = tensors
        rank = max(1, min(int(rank), self.d_t))
        num_slots = max(1, int(self.config.bridge_bank_size))
        gen = torch.Generator(device="cpu").manual_seed(923_177 + int(self.config.seed) * 191 + int(rank))
        scale = 1e-2

        slot_tokens = torch.nn.Parameter(torch.randn(num_slots, self.d_t, generator=gen, dtype=torch.float32) * scale)
        q_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        k_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        v_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        hidden_proj = torch.nn.Parameter(torch.randn(rank, rank, generator=gen, dtype=torch.float32) * scale)
        out_k = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        out_v = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        optimizer = torch.optim.Adam(
            [slot_tokens, q_proj, k_proj, v_proj, hidden_proj, out_k, out_v],
            lr=float(lr),
        )

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                live_memory = torch.stack([qk, pk, qv, pv], dim=1)
                slot_memory = slot_tokens.unsqueeze(0).expand(live_memory.shape[0], -1, -1)
                memory = torch.cat([live_memory, slot_memory], dim=1)
                q_hidden = query @ q_proj
                key_hidden = torch.einsum("nmd,dr->nmr", memory, k_proj)
                value_hidden = torch.einsum("nmd,dr->nmr", memory, v_proj)
                attn_logits = torch.einsum("nr,nmr->nm", q_hidden, key_hidden) / math.sqrt(float(rank))
                attn = torch.softmax(attn_logits, dim=-1)
                context = torch.einsum("nm,nmr->nr", attn, value_hidden)
                hidden = F.gelu(context @ hidden_proj)
                pred_k = hidden @ out_k
                pred_v = hidden @ out_v
                loss = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)
                logit_pred_k = ((base_k + pred_k) * query).sum(dim=-1)
                logit_tgt_k = ((base_k + target_k) * query).sum(dim=-1)
                logit_pred_v = ((base_v + pred_v) * query).sum(dim=-1)
                logit_tgt_v = ((base_v + target_v) * query).sum(dim=-1)
                loss = loss + float(logit_weight) * (
                    F.mse_loss(logit_pred_k, logit_tgt_k) + F.mse_loss(logit_pred_v, logit_tgt_v)
                )
                if (
                    (float(prediction_distill_weight) > 0.0 or float(dynamic_prediction_weight) > 0.0)
                    and teacher_topk_log_probs is not None
                    and teacher_topk_output_rows is not None
                ):
                    teacher_logits = teacher_topk_log_probs.to(device=qk.device, dtype=torch.float32)
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)
                    teacher_rows = teacher_topk_output_rows.to(device=qk.device, dtype=torch.float32)
                    hidden_pred = 0.5 * (((base_k + pred_k) * query) + ((base_v + pred_v) * query))
                    student_logits = torch.einsum("nd,nkd->nk", hidden_pred, teacher_rows) / math.sqrt(float(self.d_t))
                    student_log_probs = torch.log_softmax(student_logits, dim=-1)
                    if float(prediction_distill_weight) > 0.0:
                        loss = loss + float(prediction_distill_weight) * F.kl_div(
                            student_log_probs,
                            teacher_probs,
                            reduction="batchmean",
                        )
                    if float(dynamic_prediction_weight) > 0.0:
                        context_hidden = 0.5 * ((qk + qv) * query)
                        context_logits = torch.einsum("nd,nkd->nk", context_hidden, teacher_rows) / math.sqrt(float(self.d_t))
                        dynamic_teacher_probs = torch.softmax(teacher_logits + context_logits, dim=-1)
                        loss = loss + float(dynamic_prediction_weight) * F.kl_div(
                            student_log_probs,
                            dynamic_teacher_probs,
                            reduction="batchmean",
                        )
                loss.backward()
                optimizer.step()

        return (
            slot_tokens.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            q_proj.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            k_proj.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            v_proj.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            hidden_proj.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            out_k.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            out_v.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
        )

    def _fit_bridge_query_module_replace(
        self,
        quantized_k: torch.Tensor,
        predicted_k: torch.Tensor,
        quantized_v: torch.Tensor,
        predicted_v: torch.Tensor,
        query_features: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
        *,
        rank: int,
        steps: int = 100,
        lr: float = 5e-2,
        logit_weight: float = 0.5,
        prediction_distill_weight: float = 0.0,
        dynamic_prediction_weight: float = 0.0,
        teacher_topk_log_probs: torch.Tensor | None = None,
        teacher_topk_output_rows: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        feature_weights_k: torch.Tensor | None = None,
        feature_weights_v: torch.Tensor | None = None,
        span_preference_weight: float = 0.0,
        span_preference_temperature: float = 1.0,
        interaction_distill_weight: float = 0.0,
        sample_prompt_ids: torch.Tensor | None = None,
        interaction_prompt_batch_size: int = 4,
        interaction_temperature: float = 1.0,
    ) -> tuple[torch.Tensor, ...]:
        tensors = (
            quantized_k.float(),
            predicted_k.float(),
            quantized_v.float(),
            predicted_v.float(),
            query_features.float(),
            target_k.float(),
            target_v.float(),
        )
        shapes = sorted({tuple(t.shape) for t in tensors})
        if len(shapes) != 1:
            raise ValueError(f"all module-replace tensors must match, got shapes {shapes}")

        qk, pk, qv, pv, query, yk, yv = tensors
        rank = max(1, min(int(rank), self.d_t))
        min_slots = (
            0
            if self.config.quantization_correction
            in {
                "bridge_ridge_qk_dynalign_query_resampler_replace",
                "bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
            }
            else 1
        )
        num_slots = max(min_slots, int(self.config.bridge_bank_size))
        gen = torch.Generator(device="cpu").manual_seed(931_771 + int(self.config.seed) * 193 + int(rank))
        scale = 1e-2

        slot_tokens = torch.nn.Parameter(torch.randn(num_slots, self.d_t, generator=gen, dtype=torch.float32) * scale)
        q_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        k_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        v_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        hidden_proj = torch.nn.Parameter(torch.randn(rank, rank, generator=gen, dtype=torch.float32) * scale)
        out_k = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        out_v = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        optimizer = torch.optim.Adam(
            [slot_tokens, q_proj, k_proj, v_proj, hidden_proj, out_k, out_v],
            lr=float(lr),
        )
        sample_w = None
        if sample_weights is not None:
            sample_w = sample_weights.float().view(-1)
            if sample_w.numel() != qk.shape[0]:
                raise ValueError(
                    "sample_weights must align with calibration samples, "
                    f"got {int(sample_w.numel())} vs {qk.shape[0]}"
                )
            sample_w = sample_w / sample_w.mean().clamp_min(1e-8)
        feature_w_k = None
        feature_w_v = None
        if feature_weights_k is not None:
            feature_w_k = feature_weights_k.float().view(-1)
            if feature_w_k.numel() != self.d_t:
                raise ValueError(
                    "feature_weights_k must align with target width, "
                    f"got {int(feature_w_k.numel())} vs {self.d_t}"
                )
            feature_w_k = feature_w_k / feature_w_k.mean().clamp_min(1e-8)
        if feature_weights_v is not None:
            feature_w_v = feature_weights_v.float().view(-1)
            if feature_w_v.numel() != self.d_t:
                raise ValueError(
                    "feature_weights_v must align with target width, "
                    f"got {int(feature_w_v.numel())} vs {self.d_t}"
                )
            feature_w_v = feature_w_v / feature_w_v.mean().clamp_min(1e-8)
        prompt_ids_cpu = None
        unique_prompt_ids = None
        if float(interaction_distill_weight) > 0.0:
            if sample_prompt_ids is None:
                raise ValueError("sample_prompt_ids are required when interaction_distill_weight > 0")
            prompt_ids_cpu = sample_prompt_ids.detach().to("cpu", dtype=torch.long).view(-1)
            if prompt_ids_cpu.numel() != qk.shape[0]:
                raise ValueError(
                    "sample_prompt_ids must align with calibration samples, "
                    f"got {int(prompt_ids_cpu.numel())} ids for {qk.shape[0]} samples"
                )
            unique_prompt_ids = torch.unique(prompt_ids_cpu)

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                live_memory = torch.stack([qk, pk, qv, pv], dim=1)
                slot_memory = slot_tokens.unsqueeze(0).expand(live_memory.shape[0], -1, -1)
                memory = torch.cat([live_memory, slot_memory], dim=1)
                q_hidden = query @ q_proj
                key_hidden = torch.einsum("nmd,dr->nmr", memory, k_proj)
                value_hidden = torch.einsum("nmd,dr->nmr", memory, v_proj)
                attn_logits = torch.einsum("nr,nmr->nm", q_hidden, key_hidden) / math.sqrt(float(rank))
                attn = torch.softmax(attn_logits, dim=-1)
                context = torch.einsum("nm,nmr->nr", attn, value_hidden)
                hidden = F.gelu(context @ hidden_proj)
                pred_k = hidden @ out_k
                pred_v = hidden @ out_v
                diff_k = (pred_k - yk).pow(2)
                diff_v = (pred_v - yv).pow(2)
                if feature_w_k is not None:
                    diff_k = diff_k * feature_w_k.to(device=pred_k.device, dtype=pred_k.dtype).view(1, -1)
                if feature_w_v is not None:
                    diff_v = diff_v * feature_w_v.to(device=pred_v.device, dtype=pred_v.dtype).view(1, -1)
                if sample_w is None:
                    loss = diff_k.mean() + diff_v.mean()
                else:
                    weights = sample_w.to(device=pred_k.device, dtype=pred_k.dtype).view(-1, 1)
                    loss = (weights * diff_k).mean() + (weights * diff_v).mean()
                logit_pred_k = (pred_k * query).sum(dim=-1)
                logit_tgt_k = (yk * query).sum(dim=-1)
                logit_pred_v = (pred_v * query).sum(dim=-1)
                logit_tgt_v = (yv * query).sum(dim=-1)
                if sample_w is None:
                    logit_loss = F.mse_loss(logit_pred_k, logit_tgt_k) + F.mse_loss(logit_pred_v, logit_tgt_v)
                else:
                    weights_1d = sample_w.to(device=pred_k.device, dtype=pred_k.dtype)
                    logit_loss = (
                        (weights_1d * (logit_pred_k - logit_tgt_k).pow(2)).mean()
                        + (weights_1d * (logit_pred_v - logit_tgt_v).pow(2)).mean()
                    )
                loss = loss + float(logit_weight) * logit_loss
                if (
                    (
                        float(prediction_distill_weight) > 0.0
                        or float(dynamic_prediction_weight) > 0.0
                        or float(span_preference_weight) > 0.0
                    )
                    and teacher_topk_log_probs is not None
                    and teacher_topk_output_rows is not None
                ):
                    teacher_logits = teacher_topk_log_probs.to(device=qk.device, dtype=torch.float32)
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)
                    teacher_rows = teacher_topk_output_rows.to(device=qk.device, dtype=torch.float32)
                    hidden_pred = 0.5 * ((pred_k * query) + (pred_v * query))
                    student_logits = torch.einsum("nd,nkd->nk", hidden_pred, teacher_rows) / math.sqrt(float(self.d_t))
                    student_log_probs = torch.log_softmax(student_logits, dim=-1)
                    per_sample_kl = torch.sum(
                        teacher_probs * (teacher_logits - student_log_probs),
                        dim=-1,
                    )
                    if float(prediction_distill_weight) > 0.0:
                        if sample_w is None:
                            loss = loss + float(prediction_distill_weight) * per_sample_kl.mean()
                        else:
                            weights_1d = sample_w.to(device=pred_k.device, dtype=pred_k.dtype)
                            loss = loss + float(prediction_distill_weight) * (weights_1d * per_sample_kl).mean()
                    if float(dynamic_prediction_weight) > 0.0:
                        context_hidden = 0.5 * ((qk + qv) * query)
                        context_logits = torch.einsum("nd,nkd->nk", context_hidden, teacher_rows) / math.sqrt(float(self.d_t))
                        dynamic_teacher_probs = torch.softmax(teacher_logits + context_logits, dim=-1)
                        dynamic_per_sample_kl = torch.sum(
                            dynamic_teacher_probs
                            * (torch.log(dynamic_teacher_probs.clamp_min(1e-30)) - student_log_probs),
                            dim=-1,
                        )
                        if sample_w is None:
                            loss = loss + float(dynamic_prediction_weight) * dynamic_per_sample_kl.mean()
                        else:
                            weights_1d = sample_w.to(device=pred_k.device, dtype=pred_k.dtype)
                            loss = loss + float(dynamic_prediction_weight) * (weights_1d * dynamic_per_sample_kl).mean()
                    if float(span_preference_weight) > 0.0:
                        temperature = max(float(span_preference_temperature), 1e-4)
                        k = int(student_logits.shape[-1])
                        if k > 1:
                            pair_mask = torch.triu(
                                torch.ones(k, k, device=student_logits.device, dtype=torch.bool),
                                diagonal=1,
                            )
                            teacher_pair_logits = (teacher_logits.unsqueeze(-1) - teacher_logits.unsqueeze(-2)) / temperature
                            student_pair_logits = (student_logits.unsqueeze(-1) - student_logits.unsqueeze(-2)) / temperature
                            teacher_pair_probs = torch.sigmoid(teacher_pair_logits[:, pair_mask]).detach()
                            per_sample_pref = F.binary_cross_entropy_with_logits(
                                student_pair_logits[:, pair_mask],
                                teacher_pair_probs,
                                reduction="none",
                            ).mean(dim=-1)
                            if sample_w is None:
                                loss = loss + float(span_preference_weight) * per_sample_pref.mean()
                            else:
                                weights_1d = sample_w.to(device=pred_k.device, dtype=pred_k.dtype)
                                loss = loss + float(span_preference_weight) * (weights_1d * per_sample_pref).mean()
                if float(interaction_distill_weight) > 0.0 and unique_prompt_ids is not None and prompt_ids_cpu is not None:
                    if unique_prompt_ids.numel() > int(interaction_prompt_batch_size):
                        prompt_choice = unique_prompt_ids[
                            torch.randperm(unique_prompt_ids.numel(), generator=gen)[: int(interaction_prompt_batch_size)]
                        ]
                    else:
                        prompt_choice = unique_prompt_ids
                    interaction_losses: list[torch.Tensor] = []
                    temperature = max(float(interaction_temperature), 1e-4)
                    for prompt_id in prompt_choice.tolist():
                        prompt_idx = torch.nonzero(prompt_ids_cpu == int(prompt_id), as_tuple=False).view(-1)
                        if prompt_idx.numel() <= 1:
                            continue
                        prompt_idx = prompt_idx.to(device=qk.device)
                        full_pred = 0.5 * (pred_k[prompt_idx] + pred_v[prompt_idx])
                        full_tgt = 0.5 * (yk[prompt_idx] + yv[prompt_idx])
                        feat_pred = F.normalize(full_pred, dim=-1)
                        feat_tgt = F.normalize(full_tgt, dim=-1)
                        sim_pred = (feat_pred @ feat_pred.T) / temperature
                        sim_tgt = (feat_tgt @ feat_tgt.T) / temperature
                        eye_mask = torch.eye(sim_pred.shape[0], device=sim_pred.device, dtype=torch.bool)
                        sim_pred = sim_pred.masked_fill(eye_mask, -1e9)
                        sim_tgt = sim_tgt.masked_fill(eye_mask, -1e9)
                        tgt_probs = torch.softmax(sim_tgt, dim=-1)
                        pred_log_probs = torch.log_softmax(sim_pred, dim=-1)
                        interaction_losses.append(F.kl_div(pred_log_probs, tgt_probs, reduction="batchmean"))
                    if interaction_losses:
                        loss = loss + float(interaction_distill_weight) * torch.stack(interaction_losses).mean()
                loss.backward()
                optimizer.step()

        return (
            slot_tokens.detach().to(dtype=target_k.dtype, device=target_k.device),
            q_proj.detach().to(dtype=target_k.dtype, device=target_k.device),
            k_proj.detach().to(dtype=target_k.dtype, device=target_k.device),
            v_proj.detach().to(dtype=target_k.dtype, device=target_k.device),
            hidden_proj.detach().to(dtype=target_k.dtype, device=target_k.device),
            out_k.detach().to(dtype=target_k.dtype, device=target_k.device),
            out_v.detach().to(dtype=target_v.dtype, device=target_v.device),
        )

    def _predict_bridge_query_module_replace(
        self,
        quantized_k: torch.Tensor,
        predicted_k: torch.Tensor,
        quantized_v: torch.Tensor,
        predicted_v: torch.Tensor,
        query_features: torch.Tensor,
        slot_tokens: torch.Tensor,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        hidden_proj: torch.Tensor,
        out_k: torch.Tensor,
        out_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qk = quantized_k.float()
        pk = predicted_k.float()
        qv = quantized_v.float()
        pv = predicted_v.float()
        query = query_features.float()
        memory = torch.cat(
            [
                torch.stack([qk, pk, qv, pv], dim=1),
                slot_tokens.float().unsqueeze(0).expand(qk.shape[0], -1, -1),
            ],
            dim=1,
        )
        rank = max(1, int(q_proj.shape[-1]))
        q_hidden = query @ q_proj.float()
        key_hidden = torch.einsum("nmd,dr->nmr", memory, k_proj.float())
        value_hidden = torch.einsum("nmd,dr->nmr", memory, v_proj.float())
        attn_logits = torch.einsum("nr,nmr->nm", q_hidden, key_hidden) / math.sqrt(float(rank))
        attn = torch.softmax(attn_logits, dim=-1)
        context = torch.einsum("nm,nmr->nr", attn, value_hidden)
        hidden = F.gelu(context @ hidden_proj.float())
        pred_k = hidden @ out_k.float()
        pred_v = hidden @ out_v.float()
        return (
            pred_k.to(dtype=quantized_k.dtype, device=quantized_k.device),
            pred_v.to(dtype=quantized_v.dtype, device=quantized_v.device),
        )

    def _fit_bridge_query_route_gate(
        self,
        query_features: torch.Tensor,
        base_prediction: torch.Tensor,
        module_prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        steps: int = 100,
        lr: float = 5e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query_features.float()
        base = base_prediction.float()
        module = module_prediction.float()
        y = target.float()
        base_err = (base - y).pow(2).mean(dim=-1)
        module_err = (module - y).pow(2).mean(dim=-1)
        margin = base_err - module_err
        labels = (margin > 0).float()
        weights = margin.abs()
        if float(weights.mean().item()) <= 1e-8:
            weights = torch.ones_like(weights)
        weights = weights / weights.mean().clamp_min(1e-8)

        route = torch.nn.Parameter(torch.zeros(self.d_t, 1, dtype=torch.float32))
        bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        optimizer = torch.optim.Adam([route, bias], lr=float(lr))

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                logits = query @ route + bias
                gate = torch.sigmoid(logits)
                mixed = base + gate * (module - base)
                mix_loss = (weights * (mixed - y).pow(2).mean(dim=-1)).mean()
                route_loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    labels,
                    weight=weights,
                )
                loss = mix_loss + 0.5 * route_loss
                loss.backward()
                optimizer.step()

        return (
            route.detach().to(dtype=target.dtype, device=target.device),
            bias.detach().to(dtype=target.dtype, device=target.device),
        )

    def _fit_bridge_query_tokenbasis_replace(
        self,
        quantized_k: torch.Tensor,
        predicted_k: torch.Tensor,
        quantized_v: torch.Tensor,
        predicted_v: torch.Tensor,
        query_features: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
        *,
        rank: int,
        steps: int = 100,
        lr: float = 5e-2,
        logit_weight: float = 0.5,
        prediction_distill_weight: float = 0.0,
        dynamic_prediction_weight: float = 0.0,
        teacher_topk_log_probs: torch.Tensor | None = None,
        teacher_topk_output_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if teacher_topk_output_rows is None:
            raise ValueError("bridge_ridge_qk_tokenbasis_replace requires teacher_topk_output_rows")
        tensors = (
            quantized_k.float(),
            predicted_k.float(),
            quantized_v.float(),
            predicted_v.float(),
            query_features.float(),
            target_k.float(),
            target_v.float(),
        )
        shapes = sorted({tuple(t.shape) for t in tensors})
        if len(shapes) != 1:
            raise ValueError(f"all tokenbasis-replace tensors must match, got shapes {shapes}")

        qk, pk, qv, pv, query, yk, yv = tensors
        rank = max(1, min(int(rank), self.d_t))
        num_slots = max(1, int(self.config.bridge_bank_size))
        teacher_rows = teacher_topk_output_rows.to(device=qk.device, dtype=torch.float32).reshape(-1, self.d_t)
        if teacher_rows.shape[0] < rank:
            raise ValueError(
                f"teacher_topk_output_rows only has {teacher_rows.shape[0]} flattened rows, need at least rank={rank}"
            )
        basis_cols = self._top_feature_basis(teacher_rows, rank)
        token_basis = basis_cols.T.contiguous().to(dtype=torch.float32)

        gen = torch.Generator(device="cpu").manual_seed(941_771 + int(self.config.seed) * 197 + int(rank))
        scale = 1e-2
        slot_tokens = torch.nn.Parameter(torch.randn(num_slots, self.d_t, generator=gen, dtype=torch.float32) * scale)
        q_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        k_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        v_proj = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        hidden_proj = torch.nn.Parameter(torch.randn(rank, rank, generator=gen, dtype=torch.float32) * scale)
        coeff_k = torch.nn.Parameter(torch.randn(rank, rank, generator=gen, dtype=torch.float32) * scale)
        coeff_v = torch.nn.Parameter(torch.randn(rank, rank, generator=gen, dtype=torch.float32) * scale)
        optimizer = torch.optim.Adam(
            [slot_tokens, q_proj, k_proj, v_proj, hidden_proj, coeff_k, coeff_v],
            lr=float(lr),
        )

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                live_memory = torch.stack([qk, pk, qv, pv], dim=1)
                slot_memory = slot_tokens.unsqueeze(0).expand(live_memory.shape[0], -1, -1)
                memory = torch.cat([live_memory, slot_memory], dim=1)
                q_hidden = query @ q_proj
                key_hidden = torch.einsum("nmd,dr->nmr", memory, k_proj)
                value_hidden = torch.einsum("nmd,dr->nmr", memory, v_proj)
                attn_logits = torch.einsum("nr,nmr->nm", q_hidden, key_hidden) / math.sqrt(float(rank))
                attn = torch.softmax(attn_logits, dim=-1)
                context = torch.einsum("nm,nmr->nr", attn, value_hidden)
                hidden = F.gelu(context @ hidden_proj)
                pred_k = (hidden @ coeff_k) @ token_basis
                pred_v = (hidden @ coeff_v) @ token_basis
                loss = F.mse_loss(pred_k, yk) + F.mse_loss(pred_v, yv)
                logit_pred_k = (pred_k * query).sum(dim=-1)
                logit_tgt_k = (yk * query).sum(dim=-1)
                logit_pred_v = (pred_v * query).sum(dim=-1)
                logit_tgt_v = (yv * query).sum(dim=-1)
                loss = loss + float(logit_weight) * (
                    F.mse_loss(logit_pred_k, logit_tgt_k) + F.mse_loss(logit_pred_v, logit_tgt_v)
                )
                if (
                    (float(prediction_distill_weight) > 0.0 or float(dynamic_prediction_weight) > 0.0)
                    and teacher_topk_log_probs is not None
                    and teacher_topk_output_rows is not None
                ):
                    teacher_logits = teacher_topk_log_probs.to(device=qk.device, dtype=torch.float32)
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)
                    teacher_rows_now = teacher_topk_output_rows.to(device=qk.device, dtype=torch.float32)
                    hidden_pred = 0.5 * ((pred_k * query) + (pred_v * query))
                    student_logits = torch.einsum("nd,nkd->nk", hidden_pred, teacher_rows_now) / math.sqrt(float(self.d_t))
                    student_log_probs = torch.log_softmax(student_logits, dim=-1)
                    if float(prediction_distill_weight) > 0.0:
                        loss = loss + float(prediction_distill_weight) * F.kl_div(
                            student_log_probs,
                            teacher_probs,
                            reduction="batchmean",
                        )
                    if float(dynamic_prediction_weight) > 0.0:
                        context_hidden = 0.5 * ((qk + qv) * query)
                        context_logits = torch.einsum("nd,nkd->nk", context_hidden, teacher_rows_now) / math.sqrt(float(self.d_t))
                        dynamic_teacher_probs = torch.softmax(teacher_logits + context_logits, dim=-1)
                        loss = loss + float(dynamic_prediction_weight) * F.kl_div(
                            student_log_probs,
                            dynamic_teacher_probs,
                            reduction="batchmean",
                        )
                loss.backward()
                optimizer.step()

        return (
            slot_tokens.detach().to(dtype=target_k.dtype, device=target_k.device),
            q_proj.detach().to(dtype=target_k.dtype, device=target_k.device),
            k_proj.detach().to(dtype=target_k.dtype, device=target_k.device),
            v_proj.detach().to(dtype=target_k.dtype, device=target_k.device),
            hidden_proj.detach().to(dtype=target_k.dtype, device=target_k.device),
            token_basis.detach().to(dtype=target_k.dtype, device=target_k.device),
            coeff_k.detach().to(dtype=target_k.dtype, device=target_k.device),
            coeff_v.detach().to(dtype=target_v.dtype, device=target_v.device),
        )

    @staticmethod
    def _topk_sparse_codes(logits: torch.Tensor, active_k: int) -> torch.Tensor:
        active_k = max(1, min(int(active_k), logits.shape[-1]))
        relu_logits = F.relu(logits)
        if active_k >= logits.shape[-1]:
            return relu_logits
        values, indices = torch.topk(relu_logits, k=active_k, dim=-1)
        sparse = torch.zeros_like(relu_logits)
        sparse.scatter_(-1, indices, values)
        return sparse

    def _fit_bridge_query_sparse_adapter(
        self,
        quantized_k: torch.Tensor,
        predicted_k: torch.Tensor,
        quantized_v: torch.Tensor,
        predicted_v: torch.Tensor,
        query_features: torch.Tensor,
        base_prediction_k: torch.Tensor,
        base_prediction_v: torch.Tensor,
        residual_target_k: torch.Tensor,
        residual_target_v: torch.Tensor,
        *,
        rank: int,
        steps: int = 100,
        lr: float = 5e-2,
        logit_weight: float = 0.5,
        sparsity_weight: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = (
            quantized_k.float(),
            predicted_k.float(),
            quantized_v.float(),
            predicted_v.float(),
            query_features.float(),
            base_prediction_k.float(),
            base_prediction_v.float(),
            residual_target_k.float(),
            residual_target_v.float(),
        )
        shapes = sorted({tuple(t.shape) for t in tensors})
        if len(shapes) != 1:
            raise ValueError(f"all sparse adapter tensors must match, got shapes {shapes}")

        qk, pk, qv, pv, query, base_k, base_v, target_k, target_v = tensors
        rank = max(1, min(int(rank), self.d_t))
        active_k = max(1, min(2, rank))
        gen = torch.Generator(device="cpu").manual_seed(991_337 + int(self.config.seed) * 173 + int(rank))
        scale = 1e-2

        sparse_left = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        sparse_aux_left = torch.nn.Parameter(torch.randn(self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        sparse_k_right = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        sparse_v_right = torch.nn.Parameter(torch.randn(rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        optimizer = torch.optim.Adam(
            [sparse_left, sparse_aux_left, sparse_k_right, sparse_v_right],
            lr=float(lr),
        )

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                shared_query = 0.5 * ((qk + qv) * query)
                shared_aux = 0.5 * ((pk + pv) * query)
                code_logits = shared_query @ sparse_left + shared_aux @ sparse_aux_left
                codes = self._topk_sparse_codes(code_logits, active_k)
                pred_k = codes @ sparse_k_right
                pred_v = codes @ sparse_v_right
                loss = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)
                logit_pred_k = ((base_k + pred_k) * query).sum(dim=-1)
                logit_tgt_k = ((base_k + target_k) * query).sum(dim=-1)
                logit_pred_v = ((base_v + pred_v) * query).sum(dim=-1)
                logit_tgt_v = ((base_v + target_v) * query).sum(dim=-1)
                loss = loss + float(logit_weight) * (
                    F.mse_loss(logit_pred_k, logit_tgt_k) + F.mse_loss(logit_pred_v, logit_tgt_v)
                )
                loss = loss + float(sparsity_weight) * codes.abs().mean()
                loss.backward()
                optimizer.step()

        return (
            sparse_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            sparse_aux_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            sparse_k_right.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            sparse_v_right.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
        )

    def _fit_bridge_query_generated_adapter(
        self,
        quantized_k: torch.Tensor,
        predicted_k: torch.Tensor,
        quantized_v: torch.Tensor,
        predicted_v: torch.Tensor,
        query_features: torch.Tensor,
        base_prediction_k: torch.Tensor,
        base_prediction_v: torch.Tensor,
        residual_target_k: torch.Tensor,
        residual_target_v: torch.Tensor,
        *,
        rank: int,
        experts: int,
        steps: int = 100,
        lr: float = 5e-2,
        logit_weight: float = 0.5,
        batch_size: int = 512,
    ) -> tuple[torch.Tensor, ...]:
        tensors = (
            quantized_k.float(),
            predicted_k.float(),
            quantized_v.float(),
            predicted_v.float(),
            query_features.float(),
            base_prediction_k.float(),
            base_prediction_v.float(),
            residual_target_k.float(),
            residual_target_v.float(),
        )
        shapes = sorted({tuple(t.shape) for t in tensors})
        if len(shapes) != 1:
            raise ValueError(f"all generated adapter tensors must match, got shapes {shapes}")

        qk, pk, qv, pv, query, base_k, base_v, target_k, target_v = tensors
        rank = max(1, min(int(rank), self.d_t))
        experts = max(1, int(experts))
        gen = torch.Generator(device="cpu").manual_seed(771_551 + int(self.config.seed) * 181 + int(rank) * 13 + experts)
        scale = 1e-2

        hyper_left = torch.nn.Parameter(torch.randn(self.d_t, experts, generator=gen, dtype=torch.float32) * scale)
        hyper_aux_left = torch.nn.Parameter(torch.randn(self.d_t, experts, generator=gen, dtype=torch.float32) * scale)
        k_left = torch.nn.Parameter(torch.randn(experts, self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        k_right = torch.nn.Parameter(torch.randn(experts, rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        k_aux_left = torch.nn.Parameter(torch.randn(experts, self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        k_aux_right = torch.nn.Parameter(torch.randn(experts, rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        v_left = torch.nn.Parameter(torch.randn(experts, self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        v_right = torch.nn.Parameter(torch.randn(experts, rank, self.d_t, generator=gen, dtype=torch.float32) * scale)
        v_aux_left = torch.nn.Parameter(torch.randn(experts, self.d_t, rank, generator=gen, dtype=torch.float32) * scale)
        v_aux_right = torch.nn.Parameter(torch.randn(experts, rank, self.d_t, generator=gen, dtype=torch.float32) * scale)

        optimizer = torch.optim.Adam(
            [
                hyper_left,
                hyper_aux_left,
                k_left,
                k_right,
                k_aux_left,
                k_aux_right,
                v_left,
                v_right,
                v_aux_left,
                v_aux_right,
            ],
            lr=float(lr),
        )

        with torch.enable_grad():
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                if qk.shape[0] > int(batch_size):
                    batch_idx = torch.randperm(qk.shape[0], generator=gen)[: int(batch_size)].to(device=qk.device)
                else:
                    batch_idx = slice(None)
                qk_batch = qk[batch_idx]
                pk_batch = pk[batch_idx]
                qv_batch = qv[batch_idx]
                pv_batch = pv[batch_idx]
                query_batch = query[batch_idx]
                base_k_batch = base_k[batch_idx]
                base_v_batch = base_v[batch_idx]
                target_k_batch = target_k[batch_idx]
                target_v_batch = target_v[batch_idx]
                shared_query = 0.5 * ((qk_batch + qv_batch) * query_batch)
                shared_aux = 0.5 * ((pk_batch + pv_batch) * query_batch)
                coeff_logits = shared_query @ hyper_left + shared_aux @ hyper_aux_left
                coeffs = torch.softmax(
                    coeff_logits / max(float(self.config.bridge_bank_temperature), 1e-4),
                    dim=-1,
                )

                k_hidden = torch.einsum("nd,mdr->nmr", qk_batch * query_batch, k_left)
                k_aux_hidden = torch.einsum("nd,mdr->nmr", pk_batch * query_batch, k_aux_left)
                pred_k_atoms = torch.einsum("nmr,mrd->nmd", k_hidden, k_right) + torch.einsum(
                    "nmr,mrd->nmd", k_aux_hidden, k_aux_right
                )
                v_hidden = torch.einsum("nd,mdr->nmr", qv_batch * query_batch, v_left)
                v_aux_hidden = torch.einsum("nd,mdr->nmr", pv_batch * query_batch, v_aux_left)
                pred_v_atoms = torch.einsum("nmr,mrd->nmd", v_hidden, v_right) + torch.einsum(
                    "nmr,mrd->nmd", v_aux_hidden, v_aux_right
                )
                pred_k = (coeffs.unsqueeze(-1) * pred_k_atoms).sum(dim=1)
                pred_v = (coeffs.unsqueeze(-1) * pred_v_atoms).sum(dim=1)
                loss = F.mse_loss(pred_k, target_k_batch) + F.mse_loss(pred_v, target_v_batch)
                logit_pred_k = ((base_k_batch + pred_k) * query_batch).sum(dim=-1)
                logit_tgt_k = ((base_k_batch + target_k_batch) * query_batch).sum(dim=-1)
                logit_pred_v = ((base_v_batch + pred_v) * query_batch).sum(dim=-1)
                logit_tgt_v = ((base_v_batch + target_v_batch) * query_batch).sum(dim=-1)
                loss = loss + float(logit_weight) * (
                    F.mse_loss(logit_pred_k, logit_tgt_k) + F.mse_loss(logit_pred_v, logit_tgt_v)
                )
                loss.backward()
                optimizer.step()

        return (
            hyper_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            hyper_aux_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            k_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            k_right.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            k_aux_left.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            k_aux_right.detach().to(dtype=residual_target_k.dtype, device=residual_target_k.device),
            v_left.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
            v_right.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
            v_aux_left.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
            v_aux_right.detach().to(dtype=residual_target_v.dtype, device=residual_target_v.device),
        )

    def _factorize_low_rank_matrix(
        self,
        weight: torch.Tensor,
        *,
        rank: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mat = weight.float()
        max_rank = min(mat.shape[0], mat.shape[1])
        if rank is None:
            rank = min(8, max_rank)
        rank = max(1, min(int(rank), max_rank))
        try:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            jitter = 1e-6 * torch.randn_like(mat)
            u, s, vh = torch.linalg.svd(mat + jitter, full_matrices=False)
        scale = s[:rank].sqrt()
        left = u[:, :rank] * scale.unsqueeze(0)
        right = scale.unsqueeze(1) * vh[:rank, :]
        return (
            left.to(dtype=weight.dtype, device=weight.device),
            right.to(dtype=weight.dtype, device=weight.device),
        )

    def _fit_ridge_correction(
        self,
        quantized: torch.Tensor,
        target: torch.Tensor,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = quantized.float()
        y = target.float()
        q_mean = q.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        q_center = q - q_mean
        y_center = y - y_mean
        xtx = q_center.T @ q_center
        eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
        rhs = q_center.T @ y_center
        system = xtx + lam * eye
        try:
            weight = torch.linalg.solve(system, rhs)
        except RuntimeError:
            weight = torch.linalg.pinv(system) @ rhs
        bias = y_mean - q_mean @ weight
        return (
            weight.to(dtype=target.dtype, device=target.device),
            bias.to(dtype=target.dtype, device=target.device),
        )

    def _fit_low_rank_correction(
        self,
        quantized: torch.Tensor,
        target: torch.Tensor,
        *,
        rank: int | None,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = quantized.float()
        y = target.float()
        q_mean = q.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        q_center = q - q_mean
        y_center = y - y_mean
        if rank is None:
            rank = min(8, q_center.shape[1], y_center.shape[1])
        rank = max(1, min(int(rank), min(q_center.shape[1], y_center.shape[1])))
        weight = fit_alignment(
            q_center,
            y_center,
            method="reduced_rank",
            lam=lam,
            rank=rank,
        )
        bias = y_mean - q_mean @ weight
        return (
            weight.to(dtype=target.dtype, device=target.device),
            bias.to(dtype=target.dtype, device=target.device),
        )

    def _fit_coordinate_fuser(
        self,
        translated: torch.Tensor,
        target: torch.Tensor,
        *,
        dropout: float,
        salt: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dropout <= 0.0:
            zeros = torch.zeros(1, translated.shape[1], dtype=target.dtype, device=target.device)
            ones = torch.ones(1, translated.shape[1], dtype=target.dtype, device=target.device)
            return zeros, ones, zeros

        x1 = translated.float()
        y = target.float()
        gen = torch.Generator(device="cpu").manual_seed(
            131_000 + int(self.config.seed) * 1_009 + int(salt)
        )
        mask = (
            torch.rand(y.shape, generator=gen, dtype=torch.float32) >= float(dropout)
        ).to(device=y.device, dtype=y.dtype)
        x2 = y * mask
        lam = float(self.config.ridge_lambda)

        a11 = (x1 * x1).sum(dim=0) + lam
        a22 = (x2 * x2).sum(dim=0) + lam
        a12 = (x1 * x2).sum(dim=0)
        a13 = x1.sum(dim=0)
        a23 = x2.sum(dim=0)
        a33 = torch.full_like(a11, float(x1.shape[0]))
        b1 = (x1 * y).sum(dim=0)
        b2 = (x2 * y).sum(dim=0)
        b3 = y.sum(dim=0)

        mat = torch.stack(
            [
                torch.stack([a11, a12, a13], dim=-1),
                torch.stack([a12, a22, a23], dim=-1),
                torch.stack([a13, a23, a33], dim=-1),
            ],
            dim=-2,
        )
        rhs = torch.stack([b1, b2, b3], dim=-1).unsqueeze(-1)
        try:
            sol = torch.linalg.solve(mat, rhs).squeeze(-1)
        except RuntimeError:
            eye = torch.eye(3, dtype=mat.dtype, device=mat.device).unsqueeze(0)
            sol = torch.linalg.solve(mat + 1e-4 * eye, rhs).squeeze(-1)
        src_scale = sol[:, 0].unsqueeze(0)
        tgt_scale = sol[:, 1].unsqueeze(0)
        bias = sol[:, 2].unsqueeze(0)
        return (
            src_scale.to(dtype=target.dtype, device=target.device),
            tgt_scale.to(dtype=target.dtype, device=target.device),
            bias.to(dtype=target.dtype, device=target.device),
        )

    def _fit_head_ridge_fuser(
        self,
        translated: torch.Tensor,
        target: torch.Tensor,
        *,
        dropout: float,
        salt: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = translated.float()
        y = target.float()
        b, h, s, d = x1.shape
        if dropout > 0.0:
            gen = torch.Generator(device="cpu").manual_seed(
                211_000 + int(self.config.seed) * 1_009 + int(salt)
            )
            mask = (
                torch.rand(y.shape, generator=gen, dtype=torch.float32) >= float(dropout)
            ).to(device=y.device, dtype=y.dtype)
            x2 = y * mask
        else:
            x2 = y
        lam = float(self.config.ridge_lambda)
        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for head_idx in range(h):
            x1_h = x1[:, head_idx].reshape(-1, d)
            x2_h = x2[:, head_idx].reshape(-1, d)
            y_h = y[:, head_idx].reshape(-1, d)
            x_h = torch.cat([x1_h, x2_h], dim=-1)
            x_mean = x_h.mean(dim=0, keepdim=True)
            y_mean = y_h.mean(dim=0, keepdim=True)
            x_center = x_h - x_mean
            y_center = y_h - y_mean
            xtx = x_center.T @ x_center
            eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
            rhs = x_center.T @ y_center
            system = xtx + lam * eye
            try:
                weight = torch.linalg.solve(system, rhs)
            except RuntimeError:
                weight = torch.linalg.pinv(system) @ rhs
            bias = y_mean - x_mean @ weight
            weights.append(weight)
            biases.append(bias.squeeze(0))
        return (
            torch.stack(weights, dim=0).to(dtype=target.dtype, device=target.device),
            torch.stack(biases, dim=0).to(dtype=target.dtype, device=target.device),
        )

    @staticmethod
    def _js_template_distance(
        src_template: torch.Tensor,
        tgt_template: torch.Tensor,
    ) -> torch.Tensor:
        src = src_template.float().view(-1)
        tgt = tgt_template.float().view(-1)
        if src.numel() != tgt.numel():
            raise ValueError(
                "bridge templates must agree on bin count, "
                f"got {src.numel()} and {tgt.numel()}"
            )
        src = src / src.sum().clamp_min(1e-8)
        tgt = tgt / tgt.sum().clamp_min(1e-8)
        mid = 0.5 * (src + tgt)
        return 0.5 * (
            (src * (src.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum()
            + (tgt * (tgt.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum()
        )

    def set_bridge_runtime_templates(self, templates: Sequence[torch.Tensor]) -> None:
        if len(templates) != self.config.num_tgt_layers:
            raise ValueError(
                "bridge runtime template count must match target layers, "
                f"got {len(templates)} vs {self.config.num_tgt_layers}"
            )
        stacked = []
        for template in templates:
            vec = template.float().view(-1)
            if vec.numel() != self.config.transport_template_bins:
                raise ValueError(
                    "bridge runtime templates must match transport_template_bins, "
                    f"got {vec.numel()} vs {self.config.transport_template_bins}"
                )
            vec = vec / vec.sum().clamp_min(1e-8)
            stacked.append(vec)
        self.bridge_runtime_templates.copy_(
            torch.stack(stacked, dim=0).to(self.bridge_runtime_templates.device)
        )

    def _cluster_template_bank(
        self,
        template_bank: torch.Tensor,
        *,
        num_clusters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if template_bank.ndim != 2:
            raise ValueError(f"Expected [num_prompts, bins] template bank, got {tuple(template_bank.shape)}")
        if template_bank.shape[0] == 0:
            raise ValueError("template bank must contain at least one prompt")
        templates = template_bank.float()
        templates = templates / templates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        num_prompts, bins = templates.shape
        num_clusters = max(1, min(int(num_clusters), num_prompts))
        gen = torch.Generator(device="cpu").manual_seed(
            173_000 + int(self.config.seed) * 1_009 + int(num_prompts) * 17 + int(num_clusters)
        )

        def js_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            y = y / y.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            mid = 0.5 * (x + y)
            kl_x = (x * (x.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum(dim=-1)
            kl_y = (y * (y.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum(dim=-1)
            return 0.5 * (kl_x + kl_y)

        perm = torch.randperm(num_prompts, generator=gen)
        chosen = [int(perm[0].item())]
        while len(chosen) < num_clusters:
            next_idx = None
            next_dist = None
            for idx in range(num_prompts):
                if idx in chosen:
                    continue
                dist = float(js_distance(templates[idx : idx + 1], templates[chosen]).min().item())
                if next_dist is None or dist > next_dist:
                    next_dist = dist
                    next_idx = idx
            assert next_idx is not None
            chosen.append(int(next_idx))
        centroids = templates[chosen].clone()
        assignments = torch.zeros(num_prompts, dtype=torch.long)
        for _ in range(8):
            dist_mat = torch.stack(
                [js_distance(templates, centroids[cluster_idx : cluster_idx + 1]) for cluster_idx in range(num_clusters)],
                dim=1,
            )
            assignments = dist_mat.argmin(dim=1)
            updated = centroids.clone()
            for cluster_idx in range(num_clusters):
                mask = assignments == cluster_idx
                if mask.any():
                    updated[cluster_idx] = templates[mask].mean(dim=0)
                    updated[cluster_idx] = updated[cluster_idx] / updated[cluster_idx].sum().clamp_min(1e-8)
                else:
                    farthest = dist_mat.min(dim=1).values.argmax()
                    updated[cluster_idx] = templates[int(farthest.item())]
            centroids = updated
        priors = torch.bincount(assignments, minlength=num_clusters).float()
        priors = priors / priors.sum().clamp_min(1e-8)
        full_centroids = torch.zeros(num_clusters, bins, dtype=torch.float32)
        full_centroids[: centroids.shape[0]] = centroids
        return full_centroids, priors, assignments

    def _cluster_query_feature_bank(
        self,
        query_bank: torch.Tensor,
        *,
        num_clusters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_bank.ndim != 2:
            raise ValueError(f"Expected [num_samples, d_t] query bank, got {tuple(query_bank.shape)}")
        if query_bank.shape[0] == 0:
            raise ValueError("query bank must contain at least one sample")
        queries = F.normalize(query_bank.float(), dim=-1)
        num_samples, dim = queries.shape
        num_clusters = max(1, min(int(num_clusters), num_samples))
        gen = torch.Generator(device="cpu").manual_seed(
            181_000 + int(self.config.seed) * 1_009 + int(num_samples) * 17 + int(num_clusters)
        )
        perm = torch.randperm(num_samples, generator=gen)
        chosen = [int(perm[0].item())]
        while len(chosen) < num_clusters:
            next_idx = None
            next_dist = None
            for idx in range(num_samples):
                if idx in chosen:
                    continue
                dist = float(torch.cdist(queries[idx : idx + 1], queries[chosen], p=2).min().item())
                if next_dist is None or dist > next_dist:
                    next_dist = dist
                    next_idx = idx
            assert next_idx is not None
            chosen.append(int(next_idx))
        centroids = queries[chosen].clone()
        assignments = torch.zeros(num_samples, dtype=torch.long)
        for _ in range(8):
            dist_mat = torch.cdist(queries, centroids, p=2)
            assignments = dist_mat.argmin(dim=1)
            updated = centroids.clone()
            for cluster_idx in range(num_clusters):
                mask = assignments == cluster_idx
                if mask.any():
                    updated[cluster_idx] = F.normalize(queries[mask].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
                else:
                    farthest = dist_mat.min(dim=1).values.argmax()
                    updated[cluster_idx] = queries[int(farthest.item())]
            centroids = updated
        priors = torch.bincount(assignments, minlength=num_clusters).float()
        priors = priors / priors.sum().clamp_min(1e-8)
        full_centroids = torch.zeros(num_clusters, dim, dtype=torch.float32)
        full_centroids[: centroids.shape[0]] = centroids
        return full_centroids, priors, assignments

    def set_bridge_runtime_template_bank(
        self,
        template_bank: Sequence[torch.Tensor],
        sample_prompt_ids: torch.Tensor,
    ) -> None:
        if len(template_bank) != self.config.num_tgt_layers:
            raise ValueError(
                f"Expected {self.config.num_tgt_layers} bridge template banks, got {len(template_bank)}"
            )
        prompt_ids = sample_prompt_ids.detach().to("cpu", dtype=torch.long).view(-1)
        if prompt_ids.numel() == 0:
            raise ValueError("sample_prompt_ids must be non-empty")
        if int(prompt_ids.min().item()) < 0:
            raise ValueError("sample_prompt_ids must be non-negative")
        num_prompts = int(template_bank[0].shape[0])
        if int(prompt_ids.max().item()) >= num_prompts:
            raise ValueError(
                f"sample_prompt_ids reference prompt {int(prompt_ids.max().item())}, but template bank only has {num_prompts} prompts"
            )
        centroids_out: list[torch.Tensor] = []
        priors_out: list[torch.Tensor] = []
        labels_out: list[torch.Tensor] = []
        mean_templates: list[torch.Tensor] = []
        for layer_idx, bank in enumerate(template_bank):
            bank_cpu = bank.detach().to("cpu", dtype=torch.float32)
            if bank_cpu.ndim == 3 and bank_cpu.shape[1] == 1:
                bank_cpu = bank_cpu.squeeze(1)
            if bank_cpu.ndim != 2:
                raise ValueError(
                    f"Bridge template bank at layer {layer_idx} must be [num_prompts, bins], got {tuple(bank_cpu.shape)}"
                )
            if bank_cpu.shape[0] != num_prompts:
                raise ValueError(
                    f"Bridge template bank at layer {layer_idx} expected {num_prompts} prompts, got {bank_cpu.shape[0]}"
                )
            if bank_cpu.shape[1] != self.config.transport_template_bins:
                raise ValueError(
                    f"Bridge template bank at layer {layer_idx} expected {self.config.transport_template_bins} bins, got {bank_cpu.shape[1]}"
                )
            centroids, priors, assignments = self._cluster_template_bank(
                bank_cpu,
                num_clusters=self.config.bridge_bank_size,
            )
            centroids_out.append(centroids)
            priors_out.append(priors)
            labels_out.append(assignments)
            mean_templates.append(bank_cpu.mean(dim=0))
        self.bridge_bank_templates.copy_(
            torch.stack(centroids_out, dim=0).to(self.bridge_bank_templates.device)
        )
        self.bridge_bank_priors.copy_(
            torch.stack(priors_out, dim=0).to(self.bridge_bank_priors.device)
        )
        self.set_bridge_runtime_templates(mean_templates)
        self._bridge_prompt_cluster_labels = labels_out
        self._bridge_sample_prompt_ids = prompt_ids

    def set_bridge_sample_weights(self, sample_weights: Sequence[torch.Tensor]) -> None:
        if len(sample_weights) != self.config.num_tgt_layers:
            raise ValueError(
                f"Expected {self.config.num_tgt_layers} bridge sample-weight vectors, got {len(sample_weights)}"
            )
        prepared: list[torch.Tensor] = []
        for layer_idx, weights in enumerate(sample_weights):
            vec = weights.detach().to("cpu", dtype=torch.float32).view(-1)
            if vec.numel() == 0:
                raise ValueError(f"Bridge sample weights at layer {layer_idx} must be non-empty")
            if not bool(torch.isfinite(vec).all()):
                raise ValueError(f"Bridge sample weights at layer {layer_idx} must be finite")
            if float(vec.min().item()) <= 0.0:
                raise ValueError(f"Bridge sample weights at layer {layer_idx} must be positive")
            prepared.append(vec)
        self._bridge_sample_weights = prepared

    def set_bridge_sample_prompt_ids(self, sample_prompt_ids: torch.Tensor) -> None:
        prompt_ids = sample_prompt_ids.detach().to("cpu", dtype=torch.long).view(-1)
        if prompt_ids.numel() == 0:
            raise ValueError("sample_prompt_ids must be non-empty")
        if int(prompt_ids.min().item()) < 0:
            raise ValueError("sample_prompt_ids must be non-negative")
        self._bridge_sample_prompt_ids = prompt_ids

    def set_bridge_sample_query_features(self, sample_query_features: Sequence[torch.Tensor]) -> None:
        if len(sample_query_features) != self.config.num_tgt_layers:
            raise ValueError(
                f"Expected {self.config.num_tgt_layers} bridge sample-query tensors, got {len(sample_query_features)}"
            )
        prepared: list[torch.Tensor] = []
        for layer_idx, features in enumerate(sample_query_features):
            mat = features.detach().to("cpu", dtype=torch.float32)
            if mat.ndim != 2:
                raise ValueError(
                    f"Bridge sample query features at layer {layer_idx} must be rank-2 [samples, d_t], got {tuple(mat.shape)}"
                )
            if mat.shape[1] != self.d_t:
                raise ValueError(
                    f"Bridge sample query features at layer {layer_idx} expected width {self.d_t}, got {mat.shape[1]}"
                )
            if not bool(torch.isfinite(mat).all()):
                raise ValueError(f"Bridge sample query features at layer {layer_idx} must be finite")
            prepared.append(mat)
        self._bridge_sample_query_features = prepared

    def set_bridge_prediction_teacher(
        self,
        teacher_log_probs: torch.Tensor,
        teacher_output_rows: torch.Tensor,
    ) -> None:
        log_probs = teacher_log_probs.detach().to("cpu", dtype=torch.float32)
        output_rows = teacher_output_rows.detach().to("cpu", dtype=torch.float32)
        if log_probs.ndim != 2:
            raise ValueError(f"teacher_log_probs must be [samples, topk], got {tuple(log_probs.shape)}")
        if output_rows.ndim != 3:
            raise ValueError(
                f"teacher_output_rows must be [samples, topk, d_t], got {tuple(output_rows.shape)}"
            )
        if output_rows.shape[:2] != log_probs.shape:
            raise ValueError(
                "teacher_output_rows must align with teacher_log_probs, "
                f"got {tuple(output_rows.shape[:2])} vs {tuple(log_probs.shape)}"
            )
        if output_rows.shape[-1] != self.d_t:
            raise ValueError(
                f"teacher_output_rows width {output_rows.shape[-1]} does not match target width {self.d_t}"
            )
        if not bool(torch.isfinite(log_probs).all()):
            raise ValueError("teacher_log_probs must be finite")
        if not bool(torch.isfinite(output_rows).all()):
            raise ValueError("teacher_output_rows must be finite")
        self._bridge_prediction_teacher_log_probs = log_probs
        self._bridge_prediction_teacher_output_rows = output_rows

    def _bridge_runtime_gate(
        self,
        tgt_layer_idx: int,
        runtime_profile: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if runtime_profile is None:
            return torch.tensor(1.0, device=device, dtype=dtype)
        reference = self.bridge_runtime_templates[tgt_layer_idx].to(device=device, dtype=dtype)
        if float(reference.sum()) <= 1e-8:
            return torch.tensor(1.0, device=device, dtype=dtype)
        runtime = runtime_profile.to(device=device, dtype=dtype).view(-1)
        runtime = runtime / runtime.sum().clamp_min(1e-8)
        js = self._js_template_distance(runtime, reference.to(dtype=runtime.dtype))
        gate = (1.0 - js / math.log(2.0)).clamp(0.0, 1.0)
        return gate.to(device=device, dtype=dtype)

    def _bridge_bank_mixture_weights(
        self,
        tgt_layer_idx: int,
        runtime_profile: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        priors = self.bridge_bank_priors[tgt_layer_idx].to(device=device, dtype=dtype)
        active = priors > 0
        if not bool(active.any()):
            out = torch.zeros_like(priors)
            out[0] = 1.0
            return out
        if runtime_profile is None:
            return priors / priors.sum().clamp_min(1e-8)
        runtime = runtime_profile.to(device=device, dtype=dtype).view(-1)
        runtime = runtime / runtime.sum().clamp_min(1e-8)
        centroids = self.bridge_bank_templates[tgt_layer_idx].to(device=device, dtype=dtype)
        logits = []
        for expert_idx in range(self.config.bridge_bank_size):
            if not bool(active[expert_idx]):
                logits.append(torch.tensor(float("-inf"), device=device, dtype=dtype))
                continue
            centroid = centroids[expert_idx]
            centroid = centroid / centroid.sum().clamp_min(1e-8)
            js = self._js_template_distance(runtime, centroid)
            score = 1.0 - js / math.log(2.0)
            logits.append(score)
        logits_tensor = torch.stack(logits, dim=0) * float(self.config.bridge_bank_temperature)
        logits_tensor = logits_tensor + priors.clamp_min(1e-8).log()
        weights = torch.softmax(logits_tensor, dim=0)
        weights = torch.where(active, weights, torch.zeros_like(weights))
        return weights / weights.sum().clamp_min(1e-8)

    def _bridge_query_bank_mixture_weights(
        self,
        tgt_layer_idx: int,
        runtime_query_features: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        priors = self.bridge_bank_priors[tgt_layer_idx].to(device=device, dtype=dtype)
        active = priors > 0
        if not bool(active.any()):
            out = torch.zeros_like(priors)
            out[0] = 1.0
            return out
        if runtime_query_features is None:
            return priors / priors.sum().clamp_min(1e-8)
        queries = runtime_query_features.to(device=device, dtype=dtype)
        query_shape = queries.shape[:-1]
        queries = F.normalize(queries.reshape(-1, queries.shape[-1]), dim=-1)
        centroids = self.bridge_bank_query_centroids[tgt_layer_idx].to(device=device, dtype=dtype)
        centroids = F.normalize(centroids, dim=-1)
        logits = torch.full(
            (queries.shape[0], int(priors.numel())),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        active_idx = torch.nonzero(active, as_tuple=False).view(-1)
        if active_idx.numel() > 0:
            active_centroids = centroids[active_idx]
            sim = queries @ active_centroids.T
            logits[:, active_idx] = sim * float(self.config.bridge_bank_temperature)
            logits[:, active_idx] = logits[:, active_idx] + priors[active_idx].clamp_min(1e-8).log()
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(active.view(1, -1), weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return weights.view(*query_shape, -1)

    def _apply_quantization_correction(
        self,
        x: torch.Tensor,
        tgt_layer_idx: int,
        kind: str,
        *,
        aux_input: torch.Tensor | None = None,
        paired_input: torch.Tensor | None = None,
        paired_aux_input: torch.Tensor | None = None,
        runtime_profile: torch.Tensor | None = None,
        runtime_query_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.quantization_correction == "none":
            return x
        if kind == "K":
            scale = self.quant_scale_K[tgt_layer_idx]
            aux_scale = self.quant_aux_scale_K[tgt_layer_idx]
            proj = self.quant_proj_K[tgt_layer_idx]
            aux_proj = self.quant_aux_proj_K[tgt_layer_idx]
            query_proj = self.quant_query_proj_K[tgt_layer_idx]
            query_aux_proj = self.quant_query_aux_proj_K[tgt_layer_idx]
            query_resid_left = self.quant_query_resid_K_left[tgt_layer_idx]
            query_resid_right = self.quant_query_resid_K_right[tgt_layer_idx]
            query_aux_resid_left = self.quant_query_aux_resid_K_left[tgt_layer_idx]
            query_aux_resid_right = self.quant_query_aux_resid_K_right[tgt_layer_idx]
            shared_left = self.quant_query_shared_left[tgt_layer_idx]
            shared_aux_left = self.quant_query_shared_aux_left[tgt_layer_idx]
            shared_right = self.quant_query_shared_K_right[tgt_layer_idx]
            sparse_left = self.quant_query_sparse_left[tgt_layer_idx]
            sparse_aux_left = self.quant_query_sparse_aux_left[tgt_layer_idx]
            sparse_right = self.quant_query_sparse_K_right[tgt_layer_idx]
            hyper_left = self.quant_query_hyper_left[tgt_layer_idx]
            hyper_aux_left = self.quant_query_hyper_aux_left[tgt_layer_idx]
            xattn_q = self.quant_query_xattn_q[tgt_layer_idx]
            xattn_k = self.quant_query_xattn_k[tgt_layer_idx]
            xattn_v = self.quant_query_xattn_v[tgt_layer_idx]
            xattn_out = self.quant_query_xattn_K_out[tgt_layer_idx]
            module_slots = self.quant_query_module_slots[tgt_layer_idx]
            module_q = self.quant_query_module_q[tgt_layer_idx]
            module_k = self.quant_query_module_k[tgt_layer_idx]
            module_v = self.quant_query_module_v[tgt_layer_idx]
            module_hidden = self.quant_query_module_hidden[tgt_layer_idx]
            module_out = self.quant_query_module_K_out[tgt_layer_idx]
            route_proj = self.quant_query_route_K[tgt_layer_idx]
            route_bias = self.quant_query_route_K_bias[tgt_layer_idx]
            sidecar_route_proj = self.quant_query_sidecar_route_K[tgt_layer_idx]
            sidecar_route_bias = self.quant_query_sidecar_route_K_bias[tgt_layer_idx]
            token_basis = self.quant_query_token_basis[tgt_layer_idx]
            token_coeff = self.quant_query_token_K_coeff[tgt_layer_idx]
            preserve_proj = self.quant_preserve_proj_K[tgt_layer_idx]
            bias = self.quant_bias_K[tgt_layer_idx]
            bank_left = self.bridge_bank_proj_K_left[tgt_layer_idx]
            bank_right = self.bridge_bank_proj_K_right[tgt_layer_idx]
            bank_aux_left = self.bridge_bank_aux_proj_K_left[tgt_layer_idx]
            bank_aux_right = self.bridge_bank_aux_proj_K_right[tgt_layer_idx]
            bank_query_left = self.bridge_bank_query_resid_K_left[tgt_layer_idx]
            bank_query_right = self.bridge_bank_query_resid_K_right[tgt_layer_idx]
            bank_query_aux_left = self.bridge_bank_query_aux_resid_K_left[tgt_layer_idx]
            bank_query_aux_right = self.bridge_bank_query_aux_resid_K_right[tgt_layer_idx]
            bank_bias = self.bridge_bank_bias_K[tgt_layer_idx]
        elif kind == "V":
            scale = self.quant_scale_V[tgt_layer_idx]
            aux_scale = self.quant_aux_scale_V[tgt_layer_idx]
            proj = self.quant_proj_V[tgt_layer_idx]
            aux_proj = self.quant_aux_proj_V[tgt_layer_idx]
            query_proj = self.quant_query_proj_V[tgt_layer_idx]
            query_aux_proj = self.quant_query_aux_proj_V[tgt_layer_idx]
            query_resid_left = self.quant_query_resid_V_left[tgt_layer_idx]
            query_resid_right = self.quant_query_resid_V_right[tgt_layer_idx]
            query_aux_resid_left = self.quant_query_aux_resid_V_left[tgt_layer_idx]
            query_aux_resid_right = self.quant_query_aux_resid_V_right[tgt_layer_idx]
            shared_left = self.quant_query_shared_left[tgt_layer_idx]
            shared_aux_left = self.quant_query_shared_aux_left[tgt_layer_idx]
            shared_right = self.quant_query_shared_V_right[tgt_layer_idx]
            sparse_left = self.quant_query_sparse_left[tgt_layer_idx]
            sparse_aux_left = self.quant_query_sparse_aux_left[tgt_layer_idx]
            sparse_right = self.quant_query_sparse_V_right[tgt_layer_idx]
            hyper_left = self.quant_query_hyper_left[tgt_layer_idx]
            hyper_aux_left = self.quant_query_hyper_aux_left[tgt_layer_idx]
            xattn_q = self.quant_query_xattn_q[tgt_layer_idx]
            xattn_k = self.quant_query_xattn_k[tgt_layer_idx]
            xattn_v = self.quant_query_xattn_v[tgt_layer_idx]
            xattn_out = self.quant_query_xattn_V_out[tgt_layer_idx]
            module_slots = self.quant_query_module_slots[tgt_layer_idx]
            module_q = self.quant_query_module_q[tgt_layer_idx]
            module_k = self.quant_query_module_k[tgt_layer_idx]
            module_v = self.quant_query_module_v[tgt_layer_idx]
            module_hidden = self.quant_query_module_hidden[tgt_layer_idx]
            module_out = self.quant_query_module_V_out[tgt_layer_idx]
            route_proj = self.quant_query_route_V[tgt_layer_idx]
            route_bias = self.quant_query_route_V_bias[tgt_layer_idx]
            sidecar_route_proj = self.quant_query_sidecar_route_V[tgt_layer_idx]
            sidecar_route_bias = self.quant_query_sidecar_route_V_bias[tgt_layer_idx]
            token_basis = self.quant_query_token_basis[tgt_layer_idx]
            token_coeff = self.quant_query_token_V_coeff[tgt_layer_idx]
            preserve_proj = self.quant_preserve_proj_V[tgt_layer_idx]
            bias = self.quant_bias_V[tgt_layer_idx]
            bank_left = self.bridge_bank_proj_V_left[tgt_layer_idx]
            bank_right = self.bridge_bank_proj_V_right[tgt_layer_idx]
            bank_aux_left = self.bridge_bank_aux_proj_V_left[tgt_layer_idx]
            bank_aux_right = self.bridge_bank_aux_proj_V_right[tgt_layer_idx]
            bank_query_left = self.bridge_bank_query_resid_V_left[tgt_layer_idx]
            bank_query_right = self.bridge_bank_query_resid_V_right[tgt_layer_idx]
            bank_query_aux_left = self.bridge_bank_query_aux_resid_V_left[tgt_layer_idx]
            bank_query_aux_right = self.bridge_bank_query_aux_resid_V_right[tgt_layer_idx]
            bank_bias = self.bridge_bank_bias_V[tgt_layer_idx]
        else:
            raise ValueError(f"Unknown correction kind: {kind}")
        if self.config.quantization_correction == "affine":
            return x * scale.to(device=x.device, dtype=x.dtype) + bias.to(device=x.device, dtype=x.dtype)
        if self.config.quantization_correction == "bridge_affine":
            if aux_input is None:
                raise ValueError("bridge_affine quantization correction requires aux_input")
            return (
                x * scale.to(device=x.device, dtype=x.dtype)
                + aux_input * aux_scale.to(device=x.device, dtype=x.dtype)
                + bias.to(device=x.device, dtype=x.dtype)
            )
        if self.config.quantization_correction in {"bridge_ridge", "bridge_ridge_qk_weighted"}:
            if aux_input is None:
                raise ValueError(f"{self.config.quantization_correction} quantization correction requires aux_input")
            return (
                x @ proj.to(device=x.device, dtype=x.dtype)
                + aux_input @ aux_proj.to(device=x.device, dtype=x.dtype)
                + bias.to(device=x.device, dtype=x.dtype)
            )
        if self.config.quantization_correction == "bridge_ridge_qk_projector":
            if aux_input is None:
                raise ValueError("bridge_ridge_qk_projector quantization correction requires aux_input")
            if runtime_query_features is None:
                raise ValueError("bridge_ridge_qk_projector quantization correction requires runtime_query_features")
            qfeat = runtime_query_features.to(device=x.device, dtype=x.dtype)
            if x.ndim == 2:
                if qfeat.ndim == 3 and qfeat.shape[1] == 1:
                    qfeat = qfeat.squeeze(1)
                if qfeat.ndim != 2:
                    raise ValueError(
                        "runtime_query_features must have shape [samples, d_t] for calibration-time projector correction, "
                        f"got {tuple(qfeat.shape)}"
                    )
                if qfeat.shape != x.shape:
                    raise ValueError(
                        "runtime_query_features must align with calibration bridge samples, "
                        f"got {tuple(qfeat.shape)} vs {tuple(x.shape)}"
                    )
            elif qfeat.ndim == 2:
                qfeat = qfeat.unsqueeze(0)
            if x.ndim == 3 and qfeat.ndim != 3:
                raise ValueError(
                    "runtime_query_features must have shape [seq, d_t] or [batch, seq, d_t], "
                    f"got {tuple(qfeat.shape)}"
                )
            if qfeat.shape[-1] != self.d_t:
                raise ValueError(
                    f"runtime_query_features width {qfeat.shape[-1]} does not match target width {self.d_t}"
                )
            if x.ndim == 3:
                if qfeat.shape[0] == 1 and x.shape[0] != 1:
                    qfeat = qfeat.expand(x.shape[0], -1, -1)
                if qfeat.shape[1] == 1 and x.shape[1] != 1:
                    qfeat = qfeat.expand(-1, x.shape[1], -1)
                if qfeat.shape[:2] != x.shape[:2]:
                    raise ValueError(
                        "runtime_query_features must align with translated samples, "
                        f"got {tuple(qfeat.shape[:2])} vs {tuple(x.shape[:2])}"
                    )
            return (
                x @ proj.to(device=x.device, dtype=x.dtype)
                + aux_input @ aux_proj.to(device=x.device, dtype=x.dtype)
                + (x * qfeat) @ query_proj.to(device=x.device, dtype=x.dtype)
                + (aux_input * qfeat) @ query_aux_proj.to(device=x.device, dtype=x.dtype)
                + bias.to(device=x.device, dtype=x.dtype)
            )
        if self.config.quantization_correction in {"bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter", "bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter"}:
            if aux_input is None:
                raise ValueError(f"{self.config.quantization_correction} quantization correction requires aux_input")
            if runtime_query_features is None:
                raise ValueError(f"{self.config.quantization_correction} quantization correction requires runtime_query_features")
            qfeat = runtime_query_features.to(device=x.device, dtype=x.dtype)
            if x.ndim == 2:
                if qfeat.ndim == 3 and qfeat.shape[1] == 1:
                    qfeat = qfeat.squeeze(1)
                if qfeat.ndim != 2 or qfeat.shape != x.shape:
                    raise ValueError(
                        f"runtime_query_features must align with calibration bridge samples for {self.config.quantization_correction}, "
                        f"got {tuple(qfeat.shape)} vs {tuple(x.shape)}"
                    )
            else:
                if qfeat.ndim == 2:
                    qfeat = qfeat.unsqueeze(0)
                if qfeat.shape[-1] != self.d_t:
                    raise ValueError(
                        f"runtime_query_features width {qfeat.shape[-1]} does not match target width {self.d_t}"
                    )
                if qfeat.shape[0] == 1 and x.shape[0] != 1:
                    qfeat = qfeat.expand(x.shape[0], -1, -1)
                if qfeat.shape[1] == 1 and x.shape[1] != 1:
                    qfeat = qfeat.expand(-1, x.shape[1], -1)
                if qfeat.shape[:2] != x.shape[:2]:
                    raise ValueError(
                        f"runtime_query_features must align with translated samples for {self.config.quantization_correction}, "
                        f"got {tuple(qfeat.shape[:2])} vs {tuple(x.shape[:2])}"
                    )
            base = (
                x @ proj.to(device=x.device, dtype=x.dtype)
                + aux_input @ aux_proj.to(device=x.device, dtype=x.dtype)
                + bias.to(device=x.device, dtype=x.dtype)
            )
            if self.config.quantization_correction == "bridge_ridge_qk_asym_projector":
                base = base + (
                    (x * qfeat) @ query_proj.to(device=x.device, dtype=x.dtype)
                    + (aux_input * qfeat) @ query_aux_proj.to(device=x.device, dtype=x.dtype)
                )
            resid = (
                ((x * qfeat) @ query_resid_left.to(device=x.device, dtype=x.dtype))
                @ query_resid_right.to(device=x.device, dtype=x.dtype)
                + ((aux_input * qfeat) @ query_aux_resid_left.to(device=x.device, dtype=x.dtype))
                @ query_aux_resid_right.to(device=x.device, dtype=x.dtype)
            )
            if self.config.quantization_correction in {"bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter"}:
                if paired_input is None or paired_aux_input is None:
                    raise ValueError(
                        f"{self.config.quantization_correction} requires paired_input and paired_aux_input"
                    )
                shared_query = 0.5 * (
                    (x + paired_input.to(device=x.device, dtype=x.dtype)) * qfeat
                )
                shared_aux = 0.5 * (
                    (aux_input + paired_aux_input.to(device=x.device, dtype=x.dtype)) * qfeat
                )
                resid = resid + (
                    shared_query @ shared_left.to(device=x.device, dtype=x.dtype)
                    + shared_aux @ shared_aux_left.to(device=x.device, dtype=x.dtype)
                ) @ shared_right.to(device=x.device, dtype=x.dtype)
            if self.config.quantization_correction == "bridge_ridge_qk_sae_adapter":
                if paired_input is None or paired_aux_input is None:
                    raise ValueError(
                        "bridge_ridge_qk_sae_adapter requires paired_input and paired_aux_input"
                    )
                shared_query = 0.5 * (
                    (x + paired_input.to(device=x.device, dtype=x.dtype)) * qfeat
                )
                shared_aux = 0.5 * (
                    (aux_input + paired_aux_input.to(device=x.device, dtype=x.dtype)) * qfeat
                )
                code_logits = (
                    shared_query @ sparse_left.to(device=x.device, dtype=x.dtype)
                    + shared_aux @ sparse_aux_left.to(device=x.device, dtype=x.dtype)
                )
                sparse_codes = self._topk_sparse_codes(code_logits, max(1, min(2, code_logits.shape[-1])))
                resid = resid + sparse_codes @ sparse_right.to(device=x.device, dtype=x.dtype)
            if self.config.quantization_correction == "bridge_ridge_qk_generated_adapter":
                if paired_input is None or paired_aux_input is None:
                    raise ValueError(
                        "bridge_ridge_qk_generated_adapter requires paired_input and paired_aux_input"
                    )
                shared_query = 0.5 * (
                    (x + paired_input.to(device=x.device, dtype=x.dtype)) * qfeat
                )
                shared_aux = 0.5 * (
                    (aux_input + paired_aux_input.to(device=x.device, dtype=x.dtype)) * qfeat
                )
                coeff_logits = (
                    shared_query @ hyper_left.to(device=x.device, dtype=x.dtype)
                    + shared_aux @ hyper_aux_left.to(device=x.device, dtype=x.dtype)
                )
                coeffs = torch.softmax(
                    coeff_logits / max(float(self.config.bridge_bank_temperature), 1e-4),
                    dim=-1,
                )
                atom_hidden = torch.einsum(
                    "...d,mdr->...mr",
                    x * qfeat,
                    bank_query_left.to(device=x.device, dtype=x.dtype),
                )
                atom_aux_hidden = torch.einsum(
                    "...d,mdr->...mr",
                    aux_input * qfeat,
                    bank_query_aux_left.to(device=x.device, dtype=x.dtype),
                )
                atom_out = torch.einsum(
                    "...mr,mrd->...md",
                    atom_hidden,
                    bank_query_right.to(device=x.device, dtype=x.dtype),
                ) + torch.einsum(
                    "...mr,mrd->...md",
                    atom_aux_hidden,
                    bank_query_aux_right.to(device=x.device, dtype=x.dtype),
                )
                resid = resid + (coeffs.unsqueeze(-1) * atom_out).sum(dim=-2)
            if self.config.quantization_correction in {"bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter"}:
                if paired_input is None or paired_aux_input is None:
                    raise ValueError(f"{self.config.quantization_correction} requires paired_input and paired_aux_input")
                memory = torch.stack(
                    [
                        x,
                        aux_input,
                        paired_input.to(device=x.device, dtype=x.dtype),
                        paired_aux_input.to(device=x.device, dtype=x.dtype),
                    ],
                    dim=-2,
                )
                q_hidden = qfeat @ xattn_q.to(device=x.device, dtype=x.dtype)
                key_hidden = torch.einsum("...md,dr->...mr", memory, xattn_k.to(device=x.device, dtype=x.dtype))
                value_hidden = torch.einsum("...md,dr->...mr", memory, xattn_v.to(device=x.device, dtype=x.dtype))
                attn_logits = torch.einsum("...r,...mr->...m", q_hidden, key_hidden) / math.sqrt(float(max(1, xattn_q.shape[-1])))
                attn = torch.softmax(attn_logits, dim=-1)
                context = torch.einsum("...m,...mr->...r", attn, value_hidden)
                resid = resid + context @ xattn_out.to(device=x.device, dtype=x.dtype)
            if self.config.quantization_correction in {"bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace"}:
                if paired_input is None or paired_aux_input is None:
                    raise ValueError(f"{self.config.quantization_correction} requires paired_input and paired_aux_input")
                live_memory = torch.stack(
                    [
                        x,
                        aux_input,
                        paired_input.to(device=x.device, dtype=x.dtype),
                        paired_aux_input.to(device=x.device, dtype=x.dtype),
                    ],
                    dim=-2,
                )
                slot_memory = module_slots.to(device=x.device, dtype=x.dtype)
                if x.ndim == 3:
                    slot_memory = slot_memory.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
                else:
                    slot_memory = slot_memory.unsqueeze(0).expand(x.shape[0], -1, -1)
                memory = torch.cat([live_memory, slot_memory], dim=-2)
                q_hidden = qfeat @ module_q.to(device=x.device, dtype=x.dtype)
                key_hidden = torch.einsum("...md,dr->...mr", memory, module_k.to(device=x.device, dtype=x.dtype))
                value_hidden = torch.einsum("...md,dr->...mr", memory, module_v.to(device=x.device, dtype=x.dtype))
                attn_logits = torch.einsum("...r,...mr->...m", q_hidden, key_hidden) / math.sqrt(float(max(1, module_q.shape[-1])))
                attn = torch.softmax(attn_logits, dim=-1)
                context = torch.einsum("...m,...mr->...r", attn, value_hidden)
                hidden = F.gelu(context @ module_hidden.to(device=x.device, dtype=x.dtype))
                module_pred = hidden @ module_out.to(device=x.device, dtype=x.dtype)
                if self.config.quantization_correction in {"bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace"}:
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_preserve_module_replace":
                        preserve = preserve_proj.to(device=x.device, dtype=x.dtype)
                        complement = torch.eye(self.d_t, device=x.device, dtype=x.dtype) - preserve
                        return base @ preserve + module_pred @ complement
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_saliency_preserve_module_replace":
                        preserve = preserve_proj.to(device=x.device, dtype=x.dtype)
                        complement = torch.eye(self.d_t, device=x.device, dtype=x.dtype) - preserve
                        return base @ preserve + module_pred @ complement
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_anchor_tail_module_replace":
                        if kind == "K":
                            return module_pred
                        value_delta = module_pred - base
                        return base + self._quantize_tail_with_preserve(value_delta, preserve_proj)
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace":
                        if kind == "V" and tgt_layer_idx == 8:
                            value_delta = module_pred - base
                            return base + self._quantize_tail_with_preserve(value_delta, preserve_proj)
                        return module_pred
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_routed_module_replace":
                        gate = torch.sigmoid(
                            qfeat @ route_proj.to(device=x.device, dtype=x.dtype)
                            + route_bias.to(device=x.device, dtype=x.dtype)
                        )
                        return base + gate * (module_pred - base)
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_routed_module_replace":
                        if kind == "V":
                            gate = torch.sigmoid(
                                qfeat @ route_proj.to(device=x.device, dtype=x.dtype)
                                + route_bias.to(device=x.device, dtype=x.dtype)
                            )
                            return base + gate * (module_pred - base)
                        return module_pred
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_bank_module_replace":
                        if kind == "K":
                            return module_pred
                        weights = self._bridge_bank_mixture_weights(
                            tgt_layer_idx,
                            runtime_profile,
                            device=x.device,
                            dtype=x.dtype,
                        )
                        corrected = torch.zeros_like(module_pred)
                        for expert_idx in range(self.config.bridge_bank_size):
                            if float(weights[expert_idx].detach().cpu()) <= 0.0:
                                continue
                            expert_out = (
                                (x @ bank_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + (aux_input @ bank_aux_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_aux_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + bank_bias[expert_idx].to(device=x.device, dtype=x.dtype)
                            )
                            expert_out = expert_out + (
                                ((x * qfeat) @ bank_query_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_query_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + ((aux_input * qfeat) @ bank_query_aux_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_query_aux_right[expert_idx].to(device=x.device, dtype=x.dtype)
                            )
                            corrected = corrected + weights[expert_idx] * expert_out
                        return module_pred + corrected
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_query_bank_module_replace":
                        if kind == "K":
                            return module_pred
                        weights = self._bridge_query_bank_mixture_weights(
                            tgt_layer_idx,
                            qfeat,
                            device=x.device,
                            dtype=x.dtype,
                        )
                        if weights.ndim == 1:
                            weights = weights.view(*([1] * (module_pred.dim() - 1)), weights.numel())
                        corrected = torch.zeros_like(module_pred)
                        for expert_idx in range(self.config.bridge_bank_size):
                            if float(self.bridge_bank_priors[tgt_layer_idx][expert_idx].detach().cpu()) <= 0.0:
                                continue
                            expert_out = (
                                (x @ bank_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + (aux_input @ bank_aux_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_aux_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + bank_bias[expert_idx].to(device=x.device, dtype=x.dtype)
                            )
                            expert_out = expert_out + (
                                ((x * qfeat) @ bank_query_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_query_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + ((aux_input * qfeat) @ bank_query_aux_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_query_aux_right[expert_idx].to(device=x.device, dtype=x.dtype)
                            )
                            corrected = corrected + weights[..., expert_idx].unsqueeze(-1) * expert_out
                        return module_pred + corrected
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_routed_bank_module_replace":
                        if kind == "K":
                            return module_pred
                        gate = torch.sigmoid(
                            qfeat @ route_proj.to(device=x.device, dtype=x.dtype)
                            + route_bias.to(device=x.device, dtype=x.dtype)
                        )
                        weights = self._bridge_bank_mixture_weights(
                            tgt_layer_idx,
                            runtime_profile,
                            device=x.device,
                            dtype=x.dtype,
                        )
                        topk = min(2, int(weights.numel()))
                        if topk > 0 and topk < int(weights.numel()):
                            top_vals, top_idx = torch.topk(weights, k=topk)
                            sparse_weights = torch.zeros_like(weights)
                            sparse_weights[top_idx] = top_vals
                            weights = sparse_weights / sparse_weights.sum().clamp_min(1e-8)
                        corrected = torch.zeros_like(module_pred)
                        for expert_idx in range(self.config.bridge_bank_size):
                            if float(weights[expert_idx].detach().cpu()) <= 0.0:
                                continue
                            expert_out = (
                                (x @ bank_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + (aux_input @ bank_aux_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_aux_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + bank_bias[expert_idx].to(device=x.device, dtype=x.dtype)
                            )
                            expert_out = expert_out + (
                                ((x * qfeat) @ bank_query_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_query_right[expert_idx].to(device=x.device, dtype=x.dtype)
                                + ((aux_input * qfeat) @ bank_query_aux_left[expert_idx].to(device=x.device, dtype=x.dtype))
                                @ bank_query_aux_right[expert_idx].to(device=x.device, dtype=x.dtype)
                            )
                            corrected = corrected + weights[expert_idx] * expert_out
                        return base + gate * (module_pred + corrected - base)
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace":
                        if kind == "K":
                            return module_pred
                        gate = torch.sigmoid(
                            qfeat @ route_proj.to(device=x.device, dtype=x.dtype)
                            + route_bias.to(device=x.device, dtype=x.dtype)
                        )
                        routed = base + gate * (module_pred - base)
                        sidecar = (
                            ((x * qfeat) @ query_resid_left.to(device=x.device, dtype=x.dtype))
                            @ query_resid_right.to(device=x.device, dtype=x.dtype)
                            + ((aux_input * qfeat) @ query_aux_resid_left.to(device=x.device, dtype=x.dtype))
                            @ query_aux_resid_right.to(device=x.device, dtype=x.dtype)
                        )
                        sidecar_gate = torch.sigmoid(
                            qfeat @ sidecar_route_proj.to(device=x.device, dtype=x.dtype)
                            + sidecar_route_bias.to(device=x.device, dtype=x.dtype)
                        )
                        return routed + sidecar_gate * sidecar
                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_eigenspace_module_replace":
                        eigenspace = preserve_proj.to(device=x.device, dtype=x.dtype)
                        return module_pred @ eigenspace
                    return module_pred
                if self.config.quantization_correction == "bridge_ridge_qk_tokenbasis_replace":
                    coeff = hidden @ token_coeff.to(device=x.device, dtype=x.dtype)
                    return coeff @ token_basis.to(device=x.device, dtype=x.dtype)
                resid = resid + module_pred
            return base + resid
        if self.config.quantization_correction == "bridge_ridge_query":
            if aux_input is None:
                raise ValueError("bridge_ridge_query quantization correction requires aux_input")
            corrected = (
                x @ proj.to(device=x.device, dtype=x.dtype)
                + aux_input @ aux_proj.to(device=x.device, dtype=x.dtype)
                + bias.to(device=x.device, dtype=x.dtype)
            )
            gate = self._bridge_runtime_gate(
                tgt_layer_idx,
                runtime_profile,
                device=x.device,
                dtype=x.dtype,
            )
            return x + gate * (corrected - x)
        if self.config.quantization_correction in {"bridge_low_rank_bank", "bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
            if aux_input is None:
                raise ValueError(f"{self.config.quantization_correction} quantization correction requires aux_input")
            if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"} and runtime_query_features is None:
                raise ValueError(f"{self.config.quantization_correction} quantization correction requires runtime_query_features")
            base = x
            if self.config.quantization_correction in {"bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                base = (
                    x @ proj.to(device=x.device, dtype=x.dtype)
                    + aux_input @ aux_proj.to(device=x.device, dtype=x.dtype)
                    + bias.to(device=x.device, dtype=x.dtype)
                )
            weights = self._bridge_bank_mixture_weights(
                tgt_layer_idx,
                runtime_profile,
                device=x.device,
                dtype=x.dtype,
            )
            corrected = torch.zeros_like(base)
            qfeat = None
            if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                qfeat = runtime_query_features.to(device=x.device, dtype=x.dtype)
                if x.ndim == 2:
                    if qfeat.ndim == 3 and qfeat.shape[1] == 1:
                        qfeat = qfeat.squeeze(1)
                    if qfeat.ndim != 2 or qfeat.shape != x.shape:
                        raise ValueError(
                            f"runtime_query_features must align with calibration bridge samples for {self.config.quantization_correction}, "
                            f"got {tuple(qfeat.shape)} vs {tuple(x.shape)}"
                        )
                else:
                    if qfeat.ndim == 2:
                        qfeat = qfeat.unsqueeze(0)
                    if qfeat.shape[-1] != self.d_t:
                        raise ValueError(
                            f"runtime_query_features width {qfeat.shape[-1]} does not match target width {self.d_t}"
                        )
                    if qfeat.shape[0] == 1 and x.shape[0] != 1:
                        qfeat = qfeat.expand(x.shape[0], -1, -1)
                    if qfeat.shape[1] == 1 and x.shape[1] != 1:
                        qfeat = qfeat.expand(-1, x.shape[1], -1)
                    if qfeat.shape[:2] != x.shape[:2]:
                        raise ValueError(
                            f"runtime_query_features must align with translated samples for {self.config.quantization_correction}, "
                            f"got {tuple(qfeat.shape[:2])} vs {tuple(x.shape[:2])}"
                        )
            for expert_idx in range(self.config.bridge_bank_size):
                if float(weights[expert_idx].detach().cpu()) <= 0.0:
                    continue
                expert_out = (
                    (x @ bank_left[expert_idx].to(device=x.device, dtype=x.dtype))
                    @ bank_right[expert_idx].to(device=x.device, dtype=x.dtype)
                    + (aux_input @ bank_aux_left[expert_idx].to(device=x.device, dtype=x.dtype))
                    @ bank_aux_right[expert_idx].to(device=x.device, dtype=x.dtype)
                    + bank_bias[expert_idx].to(device=x.device, dtype=x.dtype)
                )
                if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"} and qfeat is not None:
                    expert_out = expert_out + (
                        ((x * qfeat) @ bank_query_left[expert_idx].to(device=x.device, dtype=x.dtype))
                        @ bank_query_right[expert_idx].to(device=x.device, dtype=x.dtype)
                        + ((aux_input * qfeat) @ bank_query_aux_left[expert_idx].to(device=x.device, dtype=x.dtype))
                        @ bank_query_aux_right[expert_idx].to(device=x.device, dtype=x.dtype)
                    )
                corrected = corrected + weights[expert_idx] * expert_out
            return corrected if self.config.quantization_correction == "bridge_low_rank_bank" else base + corrected
        if self.config.quantization_correction in {"ridge", "low_rank"}:
            return x @ proj.to(device=x.device, dtype=x.dtype) + bias.to(device=x.device, dtype=x.dtype)
        raise ValueError(f"Unknown quantization_correction: {self.config.quantization_correction}")

    # ------------------------------------------------------------------
    # Translation + fusion
    # ------------------------------------------------------------------

    def translate_layer(
        self,
        K_s: torch.Tensor,
        V_s: torch.Tensor,
        tgt_layer_idx: int,
        quantize: bool = True,
        quantization_control: str = "real",
        runtime_attention_profile: torch.Tensor | None = None,
        runtime_query_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Translate one source layer's KV into the target's KV space.

        Args:
            K_s, V_s: [batch, src_num_heads, seq, src_head_dim]
            tgt_layer_idx: target layer index l_t
            quantize: whether to round-trip through the Lloyd-Max quantizer
                      (simulates the compressed transmission channel)
            quantization_control: 'real' for true quantize/dequantize, or
                      'matched_noise' to add Gaussian noise with the same
                      empirical scale as quantization error.
        Returns:
            K_t_hat, V_t_hat: [batch, tgt_num_heads, seq, tgt_head_dim]
        """
        if not self._fitted:
            raise RuntimeError(
                "Translator has not been fit yet. Call fit_from_pairs() first."
            )

        # 1) Rotate source into its Gaussianized coordinates and flatten.
        K_s_rot = self._rotate_and_flatten(K_s, self.R_s)  # [b, s, d_s]
        V_s_rot = self._rotate_and_flatten(V_s, self.R_s)

        # 1b) Optional: apply fitted ZCA whitening to the source.
        if self._source_whitening_applies(tgt_layer_idx, "k"):
            K_s_rot = apply_whitening(
                K_s_rot,
                self.whiten_K_src[tgt_layer_idx],
                self.whiten_K_mean[tgt_layer_idx],
            )
        if self._source_whitening_applies(tgt_layer_idx, "v"):
            V_s_rot = apply_whitening(
                V_s_rot,
                self.whiten_V_src[tgt_layer_idx],
                self.whiten_V_mean[tgt_layer_idx],
            )

        # 2) Linear project into target's rotated coordinates.
        K_t_rot = K_s_rot @ self.W_K[tgt_layer_idx]  # [b, s, d_t]
        V_t_rot = V_s_rot @ self.W_V[tgt_layer_idx]

        # 2b) Optional low-rank / shrinkage filter in target space.
        K_t_rot = K_t_rot @ self.pre_quant_filter_K[tgt_layer_idx]
        V_t_rot = V_t_rot @ self.pre_quant_filter_V[tgt_layer_idx]

        # 3) Optional: round-trip through Lloyd-Max quantizer. The output is
        #    the dequantized reconstruction — bit-accurate simulation of a
        #    compressed channel, but with a differentiable straight-through
        #    estimator if we wrap in a detach trick (omitted for clarity).
        if quantize:
            K_pred = K_t_rot
            V_pred = V_t_rot
            K_q = self.quantizer.quantize_dequantize(K_t_rot)
            V_q = self.quantizer.quantize_dequantize(V_t_rot)
            if quantization_control == "real":
                K_t_rot = self._apply_quantization_correction(
                    K_q,
                    tgt_layer_idx,
                    "K",
                    aux_input=K_pred,
                    paired_input=V_q,
                    paired_aux_input=V_pred,
                    runtime_profile=runtime_attention_profile,
                    runtime_query_features=runtime_query_features,
                )
                V_t_rot = self._apply_quantization_correction(
                    V_q,
                    tgt_layer_idx,
                    "V",
                    aux_input=V_pred,
                    paired_input=K_q,
                    paired_aux_input=K_pred,
                    runtime_profile=runtime_attention_profile,
                    runtime_query_features=runtime_query_features,
                )
            elif quantization_control == "matched_noise":
                def add_matched_noise(x: torch.Tensor, q: torch.Tensor, salt: int) -> torch.Tensor:
                    q = self._apply_quantization_correction(
                        q,
                        tgt_layer_idx,
                        "K" if salt == 0 else "V",
                        aux_input=x,
                        paired_input=V_q if salt == 0 else K_q,
                        paired_aux_input=V_pred if salt == 0 else K_pred,
                        runtime_profile=runtime_attention_profile,
                        runtime_query_features=runtime_query_features,
                    )
                    err = (q - x).detach().float()
                    mean = float(err.mean().detach().cpu())
                    std = float(err.std().clamp_min(1e-8).detach().cpu())
                    gen = torch.Generator(device="cpu").manual_seed(
                        91_000 + int(self.config.seed) * 1_009 + int(tgt_layer_idx) * 31 + salt
                    )
                    noise = torch.randn(x.shape, generator=gen, dtype=torch.float32)
                    noise = (noise * std + mean).to(device=x.device, dtype=x.dtype)
                    return x + noise

                K_t_rot = add_matched_noise(K_t_rot, K_q, salt=0)
                V_t_rot = add_matched_noise(V_t_rot, V_q, salt=1)
            else:
                raise ValueError(f"Unknown quantization_control: {quantization_control}")

        if self._target_whitening_applies(tgt_layer_idx, "k"):
            K_t_rot = undo_whitening(
                K_t_rot,
                self.whiten_K_tgt_inv[tgt_layer_idx],
                self.whiten_K_tgt_mean[tgt_layer_idx],
            )
        if self._target_whitening_applies(tgt_layer_idx, "v"):
            V_t_rot = undo_whitening(
                V_t_rot,
                self.whiten_V_tgt_inv[tgt_layer_idx],
                self.whiten_V_tgt_mean[tgt_layer_idx],
            )

        # 4) Un-rotate and un-flatten back to [batch, tgt_num_heads, seq, tgt_head_dim].
        K_t_hat = self._unflatten_and_inverse_rotate(
            K_t_rot, self.config.tgt_num_heads, self.config.tgt_head_dim, self.R_t.T
        )
        V_t_hat = self._unflatten_and_inverse_rotate(
            V_t_rot, self.config.tgt_num_heads, self.config.tgt_head_dim, self.R_t.T
        )
        return K_t_hat, V_t_hat

    def fuse_layer(
        self,
        K_t: torch.Tensor,
        V_t: torch.Tensor,
        K_t_hat: torch.Tensor,
        V_t_hat: torch.Tensor,
        tgt_layer_idx: int,
        fusion_rule: str | None = None,
        head_gate_override_K: torch.Tensor | None = None,
        head_gate_override_V: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated fusion of target's own KV with translated source KV.

        K_final = (1 - alpha) * K_target_own + alpha * K_translated

        Shapes of the two inputs must match. For same-seq-length case this is
        a pointwise blend; for mixed-seq cases the caller is responsible for
        alignment (e.g., by concatenating along seq).
        """
        fusion_rule = self.config.fusion_rule if fusion_rule is None else fusion_rule
        if fusion_rule == "learned_affine":
            K_trans = self.apply_head_selection(K_t_hat, tgt_layer_idx)
            V_trans = self.apply_head_selection(V_t_hat, tgt_layer_idx)
            b, h, s, d = K_t.shape
            K_out = (
                K_trans * self.fusion_src_scale_K[tgt_layer_idx].view(1, h, 1, d)
                + K_t * self.fusion_tgt_scale_K[tgt_layer_idx].view(1, h, 1, d)
                + self.fusion_bias_K[tgt_layer_idx].view(1, h, 1, d)
            )
            V_out = (
                V_trans * self.fusion_src_scale_V[tgt_layer_idx].view(1, h, 1, d)
                + V_t * self.fusion_tgt_scale_V[tgt_layer_idx].view(1, h, 1, d)
                + self.fusion_bias_V[tgt_layer_idx].view(1, h, 1, d)
            )
            return K_out, V_out
        if fusion_rule == "learned_head_ridge":
            if self.fusion_head_proj_K is None or self.fusion_head_proj_V is None:
                raise RuntimeError("learned_head_ridge requires a translator fit with learned_fusion_dropout > 0")
            K_trans = self.apply_head_selection(K_t_hat, tgt_layer_idx)
            V_trans = self.apply_head_selection(V_t_hat, tgt_layer_idx)
            K_features = torch.cat([K_trans, K_t], dim=-1)
            V_features = torch.cat([V_trans, V_t], dim=-1)
            K_pred = torch.einsum(
                "bhsm,hmd->bhsd",
                K_features,
                self.fusion_head_proj_K[tgt_layer_idx].to(device=K_features.device, dtype=K_features.dtype),
            ) + self.fusion_head_bias_K[tgt_layer_idx].to(device=K_features.device, dtype=K_features.dtype).view(
                1, -1, 1, self.config.tgt_head_dim
            )
            V_pred = torch.einsum(
                "bhsm,hmd->bhsd",
                V_features,
                self.fusion_head_proj_V[tgt_layer_idx].to(device=V_features.device, dtype=V_features.dtype),
            ) + self.fusion_head_bias_V[tgt_layer_idx].to(device=V_features.device, dtype=V_features.dtype).view(
                1, -1, 1, self.config.tgt_head_dim
            )
            selected = self.head_selected_mask[tgt_layer_idx].to(device=K_t.device).view(1, -1, 1, 1)
            K_out = torch.where(selected, K_pred, K_t)
            V_out = torch.where(selected, V_pred, V_t)
            return K_out, V_out
        K_t_hat_selected = self.apply_head_selection(K_t_hat, tgt_layer_idx)
        V_t_hat_selected = self.apply_head_selection(V_t_hat, tgt_layer_idx)
        K_t_selected = self.apply_head_selection(K_t, tgt_layer_idx)
        V_t_selected = self.apply_head_selection(V_t, tgt_layer_idx)
        base_gate_k: torch.Tensor | float = torch.sigmoid(self.gate_K[tgt_layer_idx])
        base_gate_v: torch.Tensor | float = torch.sigmoid(self.gate_V[tgt_layer_idx])
        if head_gate_override_K is not None:
            base_gate_k = self._reshape_head_gate_override(head_gate_override_K, K_t)
        if head_gate_override_V is not None:
            base_gate_v = self._reshape_head_gate_override(head_gate_override_V, V_t)
        a_k = self._effective_gate(
            base_gate_k,
            K_t_selected,
            K_t_hat_selected,
            fusion_rule,
        )
        a_v = self._effective_gate(
            base_gate_v,
            V_t_selected,
            V_t_hat_selected,
            fusion_rule,
        )
        if self.config.quantization_correction == "bridge_ridge_qk_dynalign_query_innovation_resampler_replace":
            K_delta = self.apply_head_selection(K_t_hat, tgt_layer_idx, fill=torch.zeros_like(K_t))
            V_delta = self.apply_head_selection(V_t_hat, tgt_layer_idx, fill=torch.zeros_like(V_t))
            K_delta = self._bound_query_innovation_delta(K_delta, K_t)
            V_delta = self._bound_query_innovation_delta(V_delta, V_t)
            return K_t + a_k * K_delta, V_t + a_v * V_delta
        K_t_hat = self.apply_head_selection(K_t_hat, tgt_layer_idx, fill=K_t)
        V_t_hat = self.apply_head_selection(V_t_hat, tgt_layer_idx, fill=V_t)
        K_out = (1.0 - a_k) * K_t + a_k * K_t_hat
        V_out = (1.0 - a_v) * V_t + a_v * V_t_hat
        return K_out, V_out

    @staticmethod
    def _reshape_head_gate_override(head_gate_override: torch.Tensor, reference_kv: torch.Tensor) -> torch.Tensor:
        gate = head_gate_override.to(device=reference_kv.device, dtype=reference_kv.dtype)
        if gate.ndim == 1:
            gate = gate.view(1, -1, 1, 1)
        elif gate.ndim == 2 and gate.shape[0] == 1:
            gate = gate.view(1, gate.shape[1], 1, 1)
        elif gate.ndim != 4:
            raise ValueError(
                "head_gate_override must have shape [heads], [1, heads], or [1, heads, 1, 1], "
                f"got {tuple(gate.shape)}"
            )
        if gate.shape[1] != reference_kv.shape[1]:
            raise ValueError(
                f"head_gate_override head count {gate.shape[1]} does not match reference KV heads {reference_kv.shape[1]}"
            )
        return gate.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Calibration (closed-form)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_from_pairs(
        self,
        src_kvs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        tgt_kvs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        verbose: bool = False,
    ) -> dict[int, dict]:
        """Fit the per-layer alignment matrices W_K[l], W_V[l] in closed form.

        Args:
            src_kvs: list of length num_src_layers, each entry is a tuple
                     (K, V) of tensors of shape [B, src_num_heads, S, src_head_dim].
                     Typically this is the `past_key_values` from a HF forward
                     pass, converted to a list.
            tgt_kvs: analogous list of length num_tgt_layers.
            verbose: if True, print per-layer alignment quality.

        Returns:
            Dict mapping target layer index -> alignment-quality diagnostics.
        """
        assert len(src_kvs) == self.config.num_src_layers, (
            f"Got {len(src_kvs)} source-layer KVs, expected {self.config.num_src_layers}"
        )
        assert len(tgt_kvs) == self.config.num_tgt_layers

        if self.config.layer_pairing == "cka":
            self.layer_map = self._fit_cka_layer_map(src_kvs, tgt_kvs)

        grouped_alignment = self.config.alignment_method.startswith("grouped_") or (
            self.config.alignment_method in {
                "broadcast_template_transport",
                "broadcast_template_ot_transport",
                "broadcast_peak_template_ot_transport",
                "broadcast_retrieval_spectrum_ot_transport",
                "broadcast_qk_template_ot_transport",
            }
        )
        diagnostics: dict[int, dict] = {}

        for tgt_l in range(self.config.num_tgt_layers):
            src_l = self.layer_map[tgt_l]
            K_s, V_s = src_kvs[src_l]
            K_t, V_t = tgt_kvs[tgt_l]

            # Rotate + flatten both sides
            Ks_rot = self._rotate_and_flatten(K_s, self.R_s)  # [B, S, d_s]
            Vs_rot = self._rotate_and_flatten(V_s, self.R_s)
            Kt_rot = self._rotate_and_flatten(K_t, self.R_t)  # [B, S, d_t]
            Vt_rot = self._rotate_and_flatten(V_t, self.R_t)

            # Collapse [B, S, d] -> [B*S, d] as the sample dimension.
            Xk = Ks_rot.reshape(-1, self.d_s)
            Xv = Vs_rot.reshape(-1, self.d_s)
            Yk = Kt_rot.reshape(-1, self.d_t)
            Yv = Vt_rot.reshape(-1, self.d_t)

            # Optional ZCA whitening of the source (post-rotation). We only
            # whiten the *source*, since the target is what we're predicting
            # and should stay in its native scale for the alignment to be
            # interpretable.
            if self._source_whitening_applies(tgt_l, "k"):
                if grouped_alignment:
                    W_zca_k, mean_k = self._fit_grouped_whitening(Xk, use_target=False)
                else:
                    W_zca_k, mean_k = fit_zca_whitening(Xk)
                self.whiten_K_src[tgt_l].data.copy_(W_zca_k)
                self.whiten_K_mean[tgt_l].data.copy_(mean_k)
                Xk = apply_whitening(Xk, W_zca_k, mean_k)
            if self._source_whitening_applies(tgt_l, "v"):
                if grouped_alignment:
                    W_zca_v, mean_v = self._fit_grouped_whitening(Xv, use_target=False)
                else:
                    W_zca_v, mean_v = fit_zca_whitening(Xv)
                self.whiten_V_src[tgt_l].data.copy_(W_zca_v)
                self.whiten_V_mean[tgt_l].data.copy_(mean_v)
                Xv = apply_whitening(Xv, W_zca_v, mean_v)

            if self._target_whitening_applies(tgt_l, "k"):
                if grouped_alignment:
                    W_tgt_k, mean_tgt_k = self._fit_grouped_whitening(Yk, use_target=True)
                else:
                    W_tgt_k, mean_tgt_k = fit_zca_whitening(Yk)
                self.whiten_K_tgt[tgt_l].data.copy_(W_tgt_k)
                self.whiten_K_tgt_inv[tgt_l].data.copy_(torch.linalg.pinv(W_tgt_k).to(dtype=W_tgt_k.dtype))
                self.whiten_K_tgt_mean[tgt_l].data.copy_(mean_tgt_k)
                Yk_fit = apply_whitening(Yk, W_tgt_k, mean_tgt_k)
            else:
                Yk_fit = Yk
            if self._target_whitening_applies(tgt_l, "v"):
                if grouped_alignment:
                    W_tgt_v, mean_tgt_v = self._fit_grouped_whitening(Yv, use_target=True)
                else:
                    W_tgt_v, mean_tgt_v = fit_zca_whitening(Yv)
                self.whiten_V_tgt[tgt_l].data.copy_(W_tgt_v)
                self.whiten_V_tgt_inv[tgt_l].data.copy_(torch.linalg.pinv(W_tgt_v).to(dtype=W_tgt_v.dtype))
                self.whiten_V_tgt_mean[tgt_l].data.copy_(mean_tgt_v)
                Yv_fit = apply_whitening(Yv, W_tgt_v, mean_tgt_v)
            else:
                Yv_fit = Yv

            lam_k = self._fit_ridge_lambda(tgt_l, "k")
            lam_v = self._fit_ridge_lambda(tgt_l, "v")
            protected_rank_k = (
                self.config.fit_ridge_protected_rank
                if self._fit_ridge_override_applies(tgt_l, "k")
                else None
            )
            protected_rank_v = (
                self.config.fit_ridge_protected_rank
                if self._fit_ridge_override_applies(tgt_l, "v")
                else None
            )
            protected_mask_k = (
                None
                if grouped_alignment
                else self._fit_ridge_protected_output_mask(Yk_fit, tgt_l, "k")
            )
            protected_mask_v = (
                None
                if grouped_alignment
                else self._fit_ridge_protected_output_mask(Yv_fit, tgt_l, "v")
            )
            protected_lam_k = float(self.config.ridge_lambda) if protected_mask_k is not None else None
            protected_lam_v = float(self.config.ridge_lambda) if protected_mask_v is not None else None
            if grouped_alignment:
                protected_lam_k = float(self.config.ridge_lambda) if protected_rank_k is not None else None
                protected_lam_v = float(self.config.ridge_lambda) if protected_rank_v is not None else None

            if grouped_alignment:
                if self.config.alignment_method in {"grouped_transport", "grouped_permutation", "grouped_signature_transport", "grouped_subspace_transport", "grouped_canonical_transport", "grouped_adaptive_canonical_transport", "grouped_rotational_transport", "grouped_fitted_rotation_transport", "grouped_shared_basis_transport", "grouped_covariance_transport", "grouped_template_transport", "grouped_qk_retrieval_transport", "grouped_contrastive_template_transport", "grouped_template_subspace_transport"}:
                    W_K, plan_k = self._fit_group_transport_alignment(
                        Xk,
                        Yk_fit,
                        lam=lam_k,
                        protected_lam=protected_lam_k,
                        protected_output_mask=protected_mask_k,
                        protected_rank=protected_rank_k,
                        residual_rank=self.config.transport_residual_rank,
                        src_layer_idx=src_l,
                        tgt_layer_idx=tgt_l,
                    )
                    W_V, plan_v = self._fit_group_transport_alignment(
                        Xv,
                        Yv_fit,
                        lam=lam_v,
                        protected_lam=protected_lam_v,
                        protected_output_mask=protected_mask_v,
                        protected_rank=protected_rank_v,
                        residual_rank=self.config.transport_residual_rank,
                        src_layer_idx=src_l,
                        tgt_layer_idx=tgt_l,
                    )
                    self.transport_plan_K[tgt_l].copy_(plan_k.to(dtype=self.transport_plan_K.dtype))
                    self.transport_plan_V[tgt_l].copy_(plan_v.to(dtype=self.transport_plan_V.dtype))
                elif self.config.alignment_method in {
                    "broadcast_template_transport",
                    "broadcast_template_ot_transport",
                    "broadcast_peak_template_ot_transport",
                    "broadcast_retrieval_spectrum_ot_transport",
                    "broadcast_qk_template_ot_transport",
                }:
                    W_K, plan_k = self._fit_broadcast_template_transport_alignment(
                        Xk,
                        Yk_fit,
                        lam=lam_k,
                        residual_rank=self.config.transport_residual_rank,
                        src_layer_idx=src_l,
                        tgt_layer_idx=tgt_l,
                        use_ot=self.config.alignment_method in {
                            "broadcast_template_ot_transport",
                            "broadcast_peak_template_ot_transport",
                            "broadcast_retrieval_spectrum_ot_transport",
                            "broadcast_qk_template_ot_transport",
                        },
                    )
                    W_V, plan_v = self._fit_broadcast_template_transport_alignment(
                        Xv,
                        Yv_fit,
                        lam=lam_v,
                        residual_rank=self.config.transport_residual_rank,
                        src_layer_idx=src_l,
                        tgt_layer_idx=tgt_l,
                        use_ot=self.config.alignment_method in {
                            "broadcast_template_ot_transport",
                            "broadcast_peak_template_ot_transport",
                            "broadcast_retrieval_spectrum_ot_transport",
                            "broadcast_qk_template_ot_transport",
                        },
                    )
                    if self._broadcast_transport_plan_K is None:
                        self._broadcast_transport_plan_K = [
                            torch.zeros(
                                self.config.src_num_heads,
                                self.config.tgt_num_heads,
                                dtype=self.transport_plan_K.dtype,
                                device=self.transport_plan_K.device,
                            )
                            for _ in range(self.config.num_tgt_layers)
                        ]
                        self._broadcast_transport_plan_V = [
                            torch.zeros(
                                self.config.src_num_heads,
                                self.config.tgt_num_heads,
                                dtype=self.transport_plan_V.dtype,
                                device=self.transport_plan_V.device,
                            )
                            for _ in range(self.config.num_tgt_layers)
                        ]
                    self._broadcast_transport_plan_K[tgt_l] = plan_k.detach().to(device=self.transport_plan_K.device, dtype=self.transport_plan_K.dtype)
                    self._broadcast_transport_plan_V[tgt_l] = plan_v.detach().to(device=self.transport_plan_V.device, dtype=self.transport_plan_V.dtype)
                else:
                    W_K = self._fit_grouped_alignment(
                        Xk,
                        Yk_fit,
                        method=self.config.alignment_method,
                        lam=lam_k,
                        rank=self.config.alignment_rank,
                    )
                    W_V = self._fit_grouped_alignment(
                        Xv,
                        Yv_fit,
                        method=self.config.alignment_method,
                        lam=lam_v,
                        rank=self.config.alignment_rank,
                    )
            else:
                W_K = self._fit_alignment_with_protected_outputs(
                    Xk,
                    Yk_fit,
                    method=self.config.alignment_method,
                    lam=lam_k,
                    protected_lam=protected_lam_k,
                    protected_output_mask=protected_mask_k,
                    rank=self.config.alignment_rank,
                )
                W_V = self._fit_alignment_with_protected_outputs(
                    Xv,
                    Yv_fit,
                    method=self.config.alignment_method,
                    lam=lam_v,
                    protected_lam=protected_lam_v,
                    protected_output_mask=protected_mask_v,
                    rank=self.config.alignment_rank,
                )
            if self.config.quantization_correction in {
                "bridge_ridge_qk_dynalign_query_resampler_replace",
                "bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
            }:
                W_K = self._guard_query_resampler_fit_tensor(W_K, torch.zeros_like(W_K))
                W_V = self._guard_query_resampler_fit_tensor(W_V, torch.zeros_like(W_V))
            self.W_K[tgt_l].data.copy_(W_K.to(self.W_K[tgt_l].dtype))
            self.W_V[tgt_l].data.copy_(W_V.to(self.W_V[tgt_l].dtype))

            # Fit optional target-space denoising before quantization.
            pre_quant_filter_k = self._fit_pre_quant_filter(Yk_fit)
            pre_quant_filter_v = self._fit_pre_quant_filter(Yv_fit)
            self.pre_quant_filter_K[tgt_l].data.copy_(pre_quant_filter_k.to(self.pre_quant_filter_K[tgt_l].dtype))
            self.pre_quant_filter_V[tgt_l].data.copy_(pre_quant_filter_v.to(self.pre_quant_filter_V[tgt_l].dtype))

            # Optional decoder-side affine correction to counter quantization bias.
            self.quant_scale_K[tgt_l].data.fill_(1.0)
            self.quant_scale_V[tgt_l].data.fill_(1.0)
            self.quant_aux_scale_K[tgt_l].data.zero_()
            self.quant_aux_scale_V[tgt_l].data.zero_()
            self.quant_proj_K[tgt_l].data.copy_(torch.eye(self.d_t, dtype=self.quant_proj_K[tgt_l].dtype))
            self.quant_proj_V[tgt_l].data.copy_(torch.eye(self.d_t, dtype=self.quant_proj_V[tgt_l].dtype))
            self.quant_aux_proj_K[tgt_l].data.zero_()
            self.quant_aux_proj_V[tgt_l].data.zero_()
            self.quant_query_proj_K[tgt_l].data.zero_()
            self.quant_query_proj_V[tgt_l].data.zero_()
            self.quant_query_aux_proj_K[tgt_l].data.zero_()
            self.quant_query_aux_proj_V[tgt_l].data.zero_()
            self.quant_query_resid_K_left[tgt_l].data.zero_()
            self.quant_query_resid_K_right[tgt_l].data.zero_()
            self.quant_query_aux_resid_K_left[tgt_l].data.zero_()
            self.quant_query_aux_resid_K_right[tgt_l].data.zero_()
            self.quant_query_resid_V_left[tgt_l].data.zero_()
            self.quant_query_resid_V_right[tgt_l].data.zero_()
            self.quant_query_aux_resid_V_left[tgt_l].data.zero_()
            self.quant_query_aux_resid_V_right[tgt_l].data.zero_()
            self.quant_query_shared_left[tgt_l].data.zero_()
            self.quant_query_shared_aux_left[tgt_l].data.zero_()
            self.quant_query_shared_K_right[tgt_l].data.zero_()
            self.quant_query_shared_V_right[tgt_l].data.zero_()
            self.quant_query_sparse_left[tgt_l].data.zero_()
            self.quant_query_sparse_aux_left[tgt_l].data.zero_()
            self.quant_query_sparse_K_right[tgt_l].data.zero_()
            self.quant_query_sparse_V_right[tgt_l].data.zero_()
            self.quant_query_hyper_left[tgt_l].data.zero_()
            self.quant_query_hyper_aux_left[tgt_l].data.zero_()
            self.quant_bias_K[tgt_l].data.zero_()
            self.quant_bias_V[tgt_l].data.zero_()
            self.bridge_bank_proj_K_left[tgt_l].data.zero_()
            self.bridge_bank_proj_K_right[tgt_l].data.zero_()
            self.bridge_bank_aux_proj_K_left[tgt_l].data.zero_()
            self.bridge_bank_aux_proj_K_right[tgt_l].data.zero_()
            self.bridge_bank_query_resid_K_left[tgt_l].data.zero_()
            self.bridge_bank_query_resid_K_right[tgt_l].data.zero_()
            self.bridge_bank_query_aux_resid_K_left[tgt_l].data.zero_()
            self.bridge_bank_query_aux_resid_K_right[tgt_l].data.zero_()
            self.bridge_bank_bias_K[tgt_l].data.zero_()
            self.bridge_bank_proj_V_left[tgt_l].data.zero_()
            self.bridge_bank_proj_V_right[tgt_l].data.zero_()
            self.bridge_bank_aux_proj_V_left[tgt_l].data.zero_()
            self.bridge_bank_aux_proj_V_right[tgt_l].data.zero_()
            self.bridge_bank_query_resid_V_left[tgt_l].data.zero_()
            self.bridge_bank_query_resid_V_right[tgt_l].data.zero_()
            self.bridge_bank_query_aux_resid_V_left[tgt_l].data.zero_()
            self.bridge_bank_query_aux_resid_V_right[tgt_l].data.zero_()
            self.bridge_bank_bias_V[tgt_l].data.zero_()
            self.quant_query_xattn_q[tgt_l].data.zero_()
            self.quant_query_xattn_k[tgt_l].data.zero_()
            self.quant_query_xattn_v[tgt_l].data.zero_()
            self.quant_query_xattn_K_out[tgt_l].data.zero_()
            self.quant_query_xattn_V_out[tgt_l].data.zero_()
            self.quant_query_module_slots[tgt_l].data.zero_()
            self.quant_query_module_q[tgt_l].data.zero_()
            self.quant_query_module_k[tgt_l].data.zero_()
            self.quant_query_module_v[tgt_l].data.zero_()
            self.quant_query_module_hidden[tgt_l].data.zero_()
            self.quant_query_module_K_out[tgt_l].data.zero_()
            self.quant_query_module_V_out[tgt_l].data.zero_()
            self.quant_query_route_K[tgt_l].data.zero_()
            self.quant_query_route_V[tgt_l].data.zero_()
            self.quant_query_route_K_bias[tgt_l].data.zero_()
            self.quant_query_route_V_bias[tgt_l].data.zero_()
            self.quant_query_sidecar_route_K[tgt_l].data.zero_()
            self.quant_query_sidecar_route_V[tgt_l].data.zero_()
            self.quant_query_sidecar_route_K_bias[tgt_l].data.zero_()
            self.quant_query_sidecar_route_V_bias[tgt_l].data.zero_()
            self.quant_query_token_basis[tgt_l].data.zero_()
            self.quant_query_token_K_coeff[tgt_l].data.zero_()
            self.quant_query_token_V_coeff[tgt_l].data.zero_()
            self.quant_preserve_proj_K[tgt_l].data.zero_()
            self.quant_preserve_proj_V[tgt_l].data.zero_()
            K_pred = (Xk @ W_K) @ pre_quant_filter_k
            V_pred = (Xv @ W_V) @ pre_quant_filter_v
            K_quant = self.quantizer.quantize_dequantize(K_pred)
            V_quant = self.quantizer.quantize_dequantize(V_pred)
            if self.config.quantization_correction == "affine":
                scale_k, bias_k = self._fit_affine_correction(K_quant, Yk_fit)
                scale_v, bias_v = self._fit_affine_correction(V_quant, Yv_fit)
                self.quant_scale_K[tgt_l].data.copy_(scale_k.to(self.quant_scale_K[tgt_l].dtype))
                self.quant_scale_V[tgt_l].data.copy_(scale_v.to(self.quant_scale_V[tgt_l].dtype))
                self.quant_bias_K[tgt_l].data.copy_(bias_k.to(self.quant_bias_K[tgt_l].dtype))
                self.quant_bias_V[tgt_l].data.copy_(bias_v.to(self.quant_bias_V[tgt_l].dtype))
            elif self.config.quantization_correction == "bridge_affine":
                scale_k, aux_scale_k, bias_k = self._fit_bridge_affine_correction(K_quant, K_pred, Yk_fit)
                scale_v, aux_scale_v, bias_v = self._fit_bridge_affine_correction(V_quant, V_pred, Yv_fit)
                self.quant_scale_K[tgt_l].data.copy_(scale_k.to(self.quant_scale_K[tgt_l].dtype))
                self.quant_scale_V[tgt_l].data.copy_(scale_v.to(self.quant_scale_V[tgt_l].dtype))
                self.quant_aux_scale_K[tgt_l].data.copy_(aux_scale_k.to(self.quant_aux_scale_K[tgt_l].dtype))
                self.quant_aux_scale_V[tgt_l].data.copy_(aux_scale_v.to(self.quant_aux_scale_V[tgt_l].dtype))
                self.quant_bias_K[tgt_l].data.copy_(bias_k.to(self.quant_bias_K[tgt_l].dtype))
                self.quant_bias_V[tgt_l].data.copy_(bias_v.to(self.quant_bias_V[tgt_l].dtype))
            elif self.config.quantization_correction in {"bridge_ridge", "bridge_ridge_query", "bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank", "bridge_ridge_qk_weighted", "bridge_ridge_qk_projector", "bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter", "bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter"}:
                sample_weights = None
                module_sample_weights = None
                dynamic_module_weight_modes = {
                    "bridge_ridge_qk_dynalign_dwakd_module_replace",
                    "bridge_ridge_qk_dynalign_likelihood_module_replace",
                    "bridge_ridge_qk_dynalign_spanalm_module_replace",
                    "bridge_ridge_qk_dynalign_prefdist_module_replace",
                    "bridge_ridge_qk_dynalign_dwainteract_module_replace",
                }
                if self.config.quantization_correction in {
                    "bridge_ridge_qk_weighted",
                    *dynamic_module_weight_modes,
                }:
                    if self._bridge_sample_weights is None:
                        raise ValueError(
                            f"{self.config.quantization_correction} requires bridge sample weights; "
                            "call set_bridge_sample_weights() before fit_from_pairs"
                        )
                    sample_weights = self._bridge_sample_weights[tgt_l].to(device=Xk.device)
                    if self.config.quantization_correction in dynamic_module_weight_modes:
                        module_sample_weights = sample_weights
                elif (
                    self.config.quantization_correction
                    == "bridge_ridge_qk_dynalign_query_innovation_resampler_replace"
                    and self._bridge_sample_weights is not None
                ):
                    module_sample_weights = self._bridge_sample_weights[tgt_l].to(device=Xk.device)
                if self.config.quantization_correction == "bridge_ridge_qk_projector":
                    if self._bridge_sample_query_features is None:
                        raise ValueError(
                            "bridge_ridge_qk_projector requires bridge sample query features; "
                            "call set_bridge_sample_query_features() before fit_from_pairs"
                        )
                    query_features = self._bridge_sample_query_features[tgt_l].to(device=Xk.device)
                    proj_k, aux_proj_k, query_proj_k, query_aux_proj_k, bias_k = self._fit_bridge_ridge_query_projector_correction(
                        K_quant,
                        K_pred,
                        query_features,
                        Yk_fit,
                        lam=self.config.ridge_lambda,
                    )
                    proj_v, aux_proj_v, query_proj_v, query_aux_proj_v, bias_v = self._fit_bridge_ridge_query_projector_correction(
                        V_quant,
                        V_pred,
                        query_features,
                        Yv_fit,
                        lam=self.config.ridge_lambda,
                    )
                    self.quant_query_proj_K[tgt_l].data.copy_(query_proj_k.to(self.quant_query_proj_K[tgt_l].dtype))
                    self.quant_query_proj_V[tgt_l].data.copy_(query_proj_v.to(self.quant_query_proj_V[tgt_l].dtype))
                    self.quant_query_aux_proj_K[tgt_l].data.copy_(query_aux_proj_k.to(self.quant_query_aux_proj_K[tgt_l].dtype))
                    self.quant_query_aux_proj_V[tgt_l].data.copy_(query_aux_proj_v.to(self.quant_query_aux_proj_V[tgt_l].dtype))
                else:
                    proj_k, aux_proj_k, bias_k = self._fit_bridge_ridge_correction(
                        K_quant,
                        K_pred,
                        Yk_fit,
                        lam=self.config.ridge_lambda,
                        sample_weights=sample_weights,
                    )
                    proj_v, aux_proj_v, bias_v = self._fit_bridge_ridge_correction(
                        V_quant,
                        V_pred,
                        Yv_fit,
                        lam=self.config.ridge_lambda,
                        sample_weights=sample_weights,
                    )
                    if self.config.quantization_correction in {"bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter", "bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter"}:
                        if self._bridge_sample_query_features is None:
                            raise ValueError(
                                f"{self.config.quantization_correction} requires bridge sample query features; "
                                "call set_bridge_sample_query_features() before fit_from_pairs"
                            )
                        query_features = self._bridge_sample_query_features[tgt_l].to(device=Xk.device)
                        base_k = K_quant @ proj_k + K_pred @ aux_proj_k + bias_k
                        resid_target_k = Yk_fit - base_k
                        base_v = V_quant @ proj_v + V_pred @ aux_proj_v + bias_v
                        resid_target_v = Yv_fit - base_v
                        sample_prompt_ids = None
                        if self.config.quantization_correction in {
                            "bridge_ridge_qk_cab_adapter",
                            "bridge_ridge_qk_emkd_adapter",
                            "bridge_ridge_qk_readout_adapter",
                            "bridge_ridge_qk_dynalign_dwainteract_module_replace",
                            "bridge_ridge_qk_dynalign_interact_module_replace",
                        }:
                            if self._bridge_sample_prompt_ids is None:
                                raise ValueError(
                                    f"{self.config.quantization_correction} requires bridge sample prompt ids; "
                                    "call set_bridge_sample_prompt_ids() before fit_from_pairs"
                                )
                            sample_prompt_ids = self._bridge_sample_prompt_ids.to(device=Xk.device)
                        teacher_log_probs = None
                        teacher_output_rows = None
                        if self.config.quantization_correction in {"bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace"}:
                            if self._bridge_prediction_teacher_log_probs is None or self._bridge_prediction_teacher_output_rows is None:
                                raise ValueError(
                                    f"{self.config.quantization_correction} requires prediction-teacher tensors; "
                                    "call set_bridge_prediction_teacher() before fit_from_pairs"
                                )
                            teacher_log_probs = self._bridge_prediction_teacher_log_probs.to(device=Xk.device)
                            teacher_output_rows = self._bridge_prediction_teacher_output_rows.to(device=Xk.device)
                        if self.config.quantization_correction == "bridge_ridge_qk_asym_projector":
                            proj_k, aux_proj_k, query_proj_k, query_aux_proj_k, bias_k = self._fit_bridge_ridge_query_projector_correction(
                                K_quant,
                                K_pred,
                                query_features,
                                Yk_fit,
                                lam=self.config.ridge_lambda,
                            )
                            proj_v, aux_proj_v, query_proj_v, query_aux_proj_v, bias_v = self._fit_bridge_ridge_query_projector_correction(
                                V_quant,
                                V_pred,
                                query_features,
                                Yv_fit,
                                lam=self.config.ridge_lambda,
                            )
                            self.quant_query_proj_K[tgt_l].data.copy_(query_proj_k.to(self.quant_query_proj_K[tgt_l].dtype))
                            self.quant_query_proj_V[tgt_l].data.copy_(query_proj_v.to(self.quant_query_proj_V[tgt_l].dtype))
                            self.quant_query_aux_proj_K[tgt_l].data.copy_(query_aux_proj_k.to(self.quant_query_aux_proj_K[tgt_l].dtype))
                            self.quant_query_aux_proj_V[tgt_l].data.copy_(query_aux_proj_v.to(self.quant_query_aux_proj_V[tgt_l].dtype))
                            base_k = (
                                K_quant @ proj_k
                                + K_pred @ aux_proj_k
                                + (K_quant * query_features) @ query_proj_k
                                + (K_pred * query_features) @ query_aux_proj_k
                                + bias_k
                            )
                            resid_target_k = Yk_fit - base_k
                            base_v = (
                                V_quant @ proj_v
                                + V_pred @ aux_proj_v
                                + (V_quant * query_features) @ query_proj_v
                                + (V_pred * query_features) @ query_aux_proj_v
                                + bias_v
                            )
                            resid_target_v = Yv_fit - base_v
                        if self.config.quantization_correction in {"bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter"}:
                            (
                                xattn_q,
                                xattn_k,
                                xattn_v,
                                xattn_k_out,
                                xattn_v_out,
                            ) = self._fit_bridge_query_xattn_adapter(
                                K_quant,
                                K_pred,
                                V_quant,
                                V_pred,
                                query_features,
                                base_k,
                                base_v,
                                resid_target_k,
                                resid_target_v,
                                rank=int(self.config.quantization_correction_rank or 8),
                                dynamic_prediction_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_xattn_dynmap_adapter" else 0.0,
                                teacher_topk_log_probs=teacher_log_probs,
                                teacher_topk_output_rows=teacher_output_rows,
                            )
                            self.quant_query_xattn_q[tgt_l].data.copy_(xattn_q.to(self.quant_query_xattn_q[tgt_l].dtype))
                            self.quant_query_xattn_k[tgt_l].data.copy_(xattn_k.to(self.quant_query_xattn_k[tgt_l].dtype))
                            self.quant_query_xattn_v[tgt_l].data.copy_(xattn_v.to(self.quant_query_xattn_v[tgt_l].dtype))
                            self.quant_query_xattn_K_out[tgt_l].data.copy_(xattn_k_out.to(self.quant_query_xattn_K_out[tgt_l].dtype))
                            self.quant_query_xattn_V_out[tgt_l].data.copy_(xattn_v_out.to(self.quant_query_xattn_V_out[tgt_l].dtype))
                        if self.config.quantization_correction == "bridge_ridge_qk_module_adapter":
                            (
                                module_slots,
                                module_q,
                                module_k,
                                module_v,
                                module_hidden,
                                module_k_out,
                                module_v_out,
                            ) = self._fit_bridge_query_module_adapter(
                                K_quant,
                                K_pred,
                                V_quant,
                                V_pred,
                                query_features,
                                base_k,
                                base_v,
                                resid_target_k,
                                resid_target_v,
                                rank=int(self.config.quantization_correction_rank or 8),
                                prediction_distill_weight=0.25,
                                teacher_topk_log_probs=teacher_log_probs,
                                teacher_topk_output_rows=teacher_output_rows,
                            )
                            self.quant_query_module_slots[tgt_l].data.copy_(module_slots.to(self.quant_query_module_slots[tgt_l].dtype))
                            self.quant_query_module_q[tgt_l].data.copy_(module_q.to(self.quant_query_module_q[tgt_l].dtype))
                            self.quant_query_module_k[tgt_l].data.copy_(module_k.to(self.quant_query_module_k[tgt_l].dtype))
                            self.quant_query_module_v[tgt_l].data.copy_(module_v.to(self.quant_query_module_v[tgt_l].dtype))
                            self.quant_query_module_hidden[tgt_l].data.copy_(module_hidden.to(self.quant_query_module_hidden[tgt_l].dtype))
                            self.quant_query_module_K_out[tgt_l].data.copy_(module_k_out.to(self.quant_query_module_K_out[tgt_l].dtype))
                            self.quant_query_module_V_out[tgt_l].data.copy_(module_v_out.to(self.quant_query_module_V_out[tgt_l].dtype))
                        if self.config.quantization_correction in {"bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace"}:
                            target_k_fit = Yk_fit
                            target_v_fit = Yv_fit
                            if self.config.quantization_correction in {"bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace"}:
                                preserve_rank = max(1, min(int(self.config.quantization_correction_rank or 8), self.d_t))
                                if self.config.quantization_correction in {"bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace"}:
                                    if self.config.quantization_correction == "bridge_ridge_qk_dynalign_anchor_tail_module_replace":
                                        preserve_proj_k = torch.zeros(
                                            self.d_t,
                                            self.d_t,
                                            dtype=Yk_fit.dtype,
                                            device=Yk_fit.device,
                                        )
                                        preserve_proj_v = self._fit_saliency_preserve_projector(
                                            Yv_fit,
                                            V_pred,
                                            query_features,
                                            preserve_rank,
                                        )
                                    elif self.config.quantization_correction == "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace":
                                        preserve_proj_k = torch.zeros(
                                            self.d_t,
                                            self.d_t,
                                            dtype=Yk_fit.dtype,
                                            device=Yk_fit.device,
                                        )
                                        preserve_proj_v = torch.zeros(
                                            self.d_t,
                                            self.d_t,
                                            dtype=Yv_fit.dtype,
                                            device=Yv_fit.device,
                                        )
                                        if tgt_l == 8:
                                            preserve_proj_v = self._fit_outlier_escrow_projector(
                                                Yv_fit,
                                                V_pred,
                                                query_features,
                                                preserve_rank,
                                            )
                                    else:
                                        preserve_proj_k = self._fit_saliency_preserve_projector(
                                            Yk_fit,
                                            K_pred,
                                            query_features,
                                            preserve_rank,
                                        )
                                        preserve_proj_v = self._fit_saliency_preserve_projector(
                                            Yv_fit,
                                            V_pred,
                                            query_features,
                                            preserve_rank,
                                        )
                                else:
                                    preserve_proj_k = self._fit_preserve_projector(Yk_fit, preserve_rank)
                                    preserve_proj_v = self._fit_preserve_projector(Yv_fit, preserve_rank)
                                self.quant_preserve_proj_K[tgt_l].data.copy_(preserve_proj_k.to(self.quant_preserve_proj_K[tgt_l].dtype))
                                self.quant_preserve_proj_V[tgt_l].data.copy_(preserve_proj_v.to(self.quant_preserve_proj_V[tgt_l].dtype))
                                if self.config.quantization_correction in {"bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace"}:
                                    eye = torch.eye(self.d_t, device=Yk_fit.device, dtype=Yk_fit.dtype)
                                    target_k_fit = Yk_fit @ (eye - preserve_proj_k.to(device=Yk_fit.device, dtype=Yk_fit.dtype))
                                    target_v_fit = Yv_fit @ (eye - preserve_proj_v.to(device=Yv_fit.device, dtype=Yv_fit.dtype))
                                elif self.config.quantization_correction == "bridge_ridge_qk_dynalign_eigenspace_module_replace":
                                    target_k_fit = Yk_fit @ preserve_proj_k.to(device=Yk_fit.device, dtype=Yk_fit.dtype)
                                    target_v_fit = Yv_fit @ preserve_proj_v.to(device=Yv_fit.device, dtype=Yv_fit.dtype)
                                elif (
                                    self.config.quantization_correction == "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace"
                                    and tgt_l == 8
                                ):
                                    preserve_v = preserve_proj_v.to(device=Yv_fit.device, dtype=Yv_fit.dtype)
                                    complement_v = torch.eye(self.d_t, device=Yv_fit.device, dtype=Yv_fit.dtype) - preserve_v
                                    target_v_fit = Yv_fit @ complement_v + V_pred.detach() @ preserve_v
                            if (
                                self.config.quantization_correction
                                == "bridge_ridge_qk_dynalign_query_innovation_resampler_replace"
                            ):
                                target_k_fit = target_k_fit - base_k
                                target_v_fit = target_v_fit - base_v
                            (
                                module_slots,
                                module_q,
                                module_k,
                                module_v,
                                module_hidden,
                                module_k_out,
                                module_v_out,
                            ) = self._fit_bridge_query_module_replace(
                                K_quant,
                                K_pred,
                                V_quant,
                                V_pred,
                                query_features,
                                target_k_fit,
                                target_v_fit,
                                rank=int(self.config.quantization_correction_rank or 8),
                                prediction_distill_weight=0.25,
                                dynamic_prediction_weight=0.25 if self.config.quantization_correction in {"bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace"} else 0.0,
                                teacher_topk_log_probs=teacher_log_probs,
                                teacher_topk_output_rows=teacher_output_rows,
                                sample_weights=module_sample_weights,
                                feature_weights_k=self._fit_saliency_feature_weights(target_k_fit, K_pred, query_features) if self.config.quantization_correction == "bridge_ridge_qk_dynalign_saliency_module_replace" else None,
                                feature_weights_v=self._fit_saliency_feature_weights(target_v_fit, V_pred, query_features) if self.config.quantization_correction == "bridge_ridge_qk_dynalign_saliency_module_replace" else None,
                                span_preference_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_dynalign_prefdist_module_replace" else 0.0,
                                interaction_distill_weight=0.25 if self.config.quantization_correction in {"bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace"} else 0.0,
                                sample_prompt_ids=sample_prompt_ids if self.config.quantization_correction in {"bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace"} else None,
                            )
                            if self.config.quantization_correction in {
                                "bridge_ridge_qk_dynalign_query_resampler_replace",
                                "bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
                            }:
                                module_slots = self._guard_query_resampler_fit_tensor(
                                    module_slots,
                                    torch.zeros_like(module_slots),
                                )
                                module_q = self._guard_query_resampler_fit_tensor(module_q, torch.zeros_like(module_q))
                                module_k = self._guard_query_resampler_fit_tensor(module_k, torch.zeros_like(module_k))
                                module_v = self._guard_query_resampler_fit_tensor(module_v, torch.zeros_like(module_v))
                                module_hidden = self._guard_query_resampler_fit_tensor(
                                    module_hidden,
                                    torch.zeros_like(module_hidden),
                                )
                                module_k_out = self._guard_query_resampler_fit_tensor(
                                    module_k_out,
                                    torch.zeros_like(module_k_out),
                                )
                                module_v_out = self._guard_query_resampler_fit_tensor(
                                    module_v_out,
                                    torch.zeros_like(module_v_out),
                                )
                            self.quant_query_module_slots[tgt_l].data.copy_(module_slots.to(self.quant_query_module_slots[tgt_l].dtype))
                            self.quant_query_module_q[tgt_l].data.copy_(module_q.to(self.quant_query_module_q[tgt_l].dtype))
                            self.quant_query_module_k[tgt_l].data.copy_(module_k.to(self.quant_query_module_k[tgt_l].dtype))
                            self.quant_query_module_v[tgt_l].data.copy_(module_v.to(self.quant_query_module_v[tgt_l].dtype))
                            self.quant_query_module_hidden[tgt_l].data.copy_(module_hidden.to(self.quant_query_module_hidden[tgt_l].dtype))
                            self.quant_query_module_K_out[tgt_l].data.copy_(module_k_out.to(self.quant_query_module_K_out[tgt_l].dtype))
                            self.quant_query_module_V_out[tgt_l].data.copy_(module_v_out.to(self.quant_query_module_V_out[tgt_l].dtype))
                            if self.config.quantization_correction == "bridge_ridge_qk_dynalign_routed_module_replace":
                                pred_fit_k, pred_fit_v = self._predict_bridge_query_module_replace(
                                    K_quant,
                                    K_pred,
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    module_slots,
                                    module_q,
                                    module_k,
                                    module_v,
                                    module_hidden,
                                    module_k_out,
                                    module_v_out,
                                )
                                route_k, route_k_bias = self._fit_bridge_query_route_gate(
                                    query_features,
                                    base_k,
                                    pred_fit_k,
                                    Yk_fit,
                                )
                                route_v, route_v_bias = self._fit_bridge_query_route_gate(
                                    query_features,
                                    base_v,
                                    pred_fit_v,
                                    Yv_fit,
                                )
                                self.quant_query_route_K[tgt_l].data.copy_(route_k.to(self.quant_query_route_K[tgt_l].dtype))
                                self.quant_query_route_V[tgt_l].data.copy_(route_v.to(self.quant_query_route_V[tgt_l].dtype))
                                self.quant_query_route_K_bias[tgt_l].data.copy_(route_k_bias.to(self.quant_query_route_K_bias[tgt_l].dtype))
                                self.quant_query_route_V_bias[tgt_l].data.copy_(route_v_bias.to(self.quant_query_route_V_bias[tgt_l].dtype))
                            if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_routed_module_replace":
                                _, pred_fit_v = self._predict_bridge_query_module_replace(
                                    K_quant,
                                    K_pred,
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    module_slots,
                                    module_q,
                                    module_k,
                                    module_v,
                                    module_hidden,
                                    module_k_out,
                                    module_v_out,
                                )
                                route_v, route_v_bias = self._fit_bridge_query_route_gate(
                                    query_features,
                                    base_v,
                                    pred_fit_v,
                                    Yv_fit,
                                )
                                self.quant_query_route_V[tgt_l].data.copy_(route_v.to(self.quant_query_route_V[tgt_l].dtype))
                                self.quant_query_route_V_bias[tgt_l].data.copy_(route_v_bias.to(self.quant_query_route_V_bias[tgt_l].dtype))
                            if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace":
                                _, pred_fit_v = self._predict_bridge_query_module_replace(
                                    K_quant,
                                    K_pred,
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    module_slots,
                                    module_q,
                                    module_k,
                                    module_v,
                                    module_hidden,
                                    module_k_out,
                                    module_v_out,
                                )
                                route_v, route_v_bias = self._fit_bridge_query_route_gate(
                                    query_features,
                                    base_v,
                                    pred_fit_v,
                                    Yv_fit,
                                )
                                self.quant_query_route_K[tgt_l].data.zero_()
                                self.quant_query_route_K_bias[tgt_l].data.zero_()
                                self.quant_query_route_V[tgt_l].data.copy_(route_v.to(self.quant_query_route_V[tgt_l].dtype))
                                self.quant_query_route_V_bias[tgt_l].data.copy_(route_v_bias.to(self.quant_query_route_V_bias[tgt_l].dtype))
                                gate_v = torch.sigmoid(
                                    query_features @ route_v.to(device=Xk.device, dtype=Xk.dtype)
                                    + route_v_bias.to(device=Xk.device, dtype=Xk.dtype)
                                )
                                routed_pred_v = base_v + gate_v * (pred_fit_v - base_v)
                                left_v, right_v, aux_left_v, aux_right_v = self._fit_bridge_query_residual_adapter(
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    routed_pred_v,
                                    Yv_fit - routed_pred_v,
                                    rank=int(self.config.quantization_correction_rank or 8),
                                )
                                self.quant_query_resid_K_left[tgt_l].data.zero_()
                                self.quant_query_resid_K_right[tgt_l].data.zero_()
                                self.quant_query_aux_resid_K_left[tgt_l].data.zero_()
                                self.quant_query_aux_resid_K_right[tgt_l].data.zero_()
                                self.quant_query_resid_V_left[tgt_l].data.copy_(left_v.to(self.quant_query_resid_V_left[tgt_l].dtype))
                                self.quant_query_resid_V_right[tgt_l].data.copy_(right_v.to(self.quant_query_resid_V_right[tgt_l].dtype))
                                self.quant_query_aux_resid_V_left[tgt_l].data.copy_(aux_left_v.to(self.quant_query_aux_resid_V_left[tgt_l].dtype))
                                self.quant_query_aux_resid_V_right[tgt_l].data.copy_(aux_right_v.to(self.quant_query_aux_resid_V_right[tgt_l].dtype))
                                sidecar_pred_v = routed_pred_v + (
                                    ((V_quant * query_features) @ left_v.to(device=Xk.device, dtype=Xk.dtype))
                                    @ right_v.to(device=Xk.device, dtype=Xk.dtype)
                                    + ((V_pred * query_features) @ aux_left_v.to(device=Xk.device, dtype=Xk.dtype))
                                    @ aux_right_v.to(device=Xk.device, dtype=Xk.dtype)
                                )
                                sidecar_route_v, sidecar_route_v_bias = self._fit_bridge_query_route_gate(
                                    query_features,
                                    routed_pred_v,
                                    sidecar_pred_v,
                                    Yv_fit,
                                )
                                self.quant_query_sidecar_route_K[tgt_l].data.zero_()
                                self.quant_query_sidecar_route_K_bias[tgt_l].data.zero_()
                                self.quant_query_sidecar_route_V[tgt_l].data.copy_(
                                    sidecar_route_v.to(self.quant_query_sidecar_route_V[tgt_l].dtype)
                                )
                                self.quant_query_sidecar_route_V_bias[tgt_l].data.copy_(
                                    sidecar_route_v_bias.to(self.quant_query_sidecar_route_V_bias[tgt_l].dtype)
                                )
                            if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_routed_bank_module_replace":
                                if self._bridge_prompt_cluster_labels is None or self._bridge_sample_prompt_ids is None:
                                    raise ValueError(
                                        "bridge_ridge_qk_dynalign_value_routed_bank_module_replace requires bridge template-bank metadata; "
                                        "call set_bridge_runtime_template_bank() before fit_from_pairs"
                                    )
                                _, pred_fit_v = self._predict_bridge_query_module_replace(
                                    K_quant,
                                    K_pred,
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    module_slots,
                                    module_q,
                                    module_k,
                                    module_v,
                                    module_hidden,
                                    module_k_out,
                                    module_v_out,
                                )
                                route_v, route_v_bias = self._fit_bridge_query_route_gate(
                                    query_features,
                                    base_v,
                                    pred_fit_v,
                                    Yv_fit,
                                )
                                self.quant_query_route_K[tgt_l].data.zero_()
                                self.quant_query_route_K_bias[tgt_l].data.zero_()
                                self.quant_query_route_V[tgt_l].data.copy_(route_v.to(self.quant_query_route_V[tgt_l].dtype))
                                self.quant_query_route_V_bias[tgt_l].data.copy_(route_v_bias.to(self.quant_query_route_V_bias[tgt_l].dtype))
                                prompt_ids = self._bridge_sample_prompt_ids.to(device=Xk.device)
                                cluster_labels = self._bridge_prompt_cluster_labels[tgt_l].to(device=Xk.device, dtype=torch.long)
                                sample_expert_ids = cluster_labels[prompt_ids]
                                gate_v = torch.sigmoid(query_features @ route_v.to(device=Xk.device, dtype=Xk.dtype) + route_v_bias.to(device=Xk.device, dtype=Xk.dtype))
                                routed_pred_v = base_v + gate_v * (pred_fit_v - base_v)
                                min_samples = max(8, int(self.config.quantization_correction_rank or 8))
                                residual_target_v = Yv_fit - routed_pred_v
                                global_left_v, global_right_v, global_aux_left_v, global_aux_right_v = self._fit_bridge_query_residual_adapter(
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    routed_pred_v,
                                    residual_target_v,
                                    rank=int(self.config.quantization_correction_rank or 8),
                                )
                                zero_bias_k = torch.zeros_like(self.bridge_bank_bias_K[tgt_l].data[0])
                                zero_bias_v = torch.zeros_like(self.bridge_bank_bias_V[tgt_l].data[0])
                                for expert_idx in range(self.config.bridge_bank_size):
                                    mask = sample_expert_ids == expert_idx
                                    if int(mask.sum().item()) >= min_samples:
                                        left_v, right_v, aux_left_v, aux_right_v = self._fit_bridge_query_residual_adapter(
                                            V_quant[mask],
                                            V_pred[mask],
                                            query_features[mask],
                                            routed_pred_v[mask],
                                            residual_target_v[mask],
                                            rank=int(self.config.quantization_correction_rank or 8),
                                        )
                                    else:
                                        left_v, right_v = global_left_v, global_right_v
                                        aux_left_v, aux_right_v = global_aux_left_v, global_aux_right_v
                                    self.bridge_bank_proj_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_proj_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_aux_resid_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_aux_resid_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_bias_K[tgt_l].data[expert_idx].copy_(zero_bias_k.to(self.bridge_bank_bias_K[tgt_l].dtype))
                                    self.bridge_bank_proj_V_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_proj_V_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_V_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_V_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_V_left[tgt_l].data[expert_idx].copy_(left_v.to(self.bridge_bank_query_resid_V_left[tgt_l].dtype))
                                    self.bridge_bank_query_resid_V_right[tgt_l].data[expert_idx].copy_(right_v.to(self.bridge_bank_query_resid_V_right[tgt_l].dtype))
                                    self.bridge_bank_query_aux_resid_V_left[tgt_l].data[expert_idx].copy_(aux_left_v.to(self.bridge_bank_query_aux_resid_V_left[tgt_l].dtype))
                                    self.bridge_bank_query_aux_resid_V_right[tgt_l].data[expert_idx].copy_(aux_right_v.to(self.bridge_bank_query_aux_resid_V_right[tgt_l].dtype))
                                    self.bridge_bank_bias_V[tgt_l].data[expert_idx].copy_(zero_bias_v.to(self.bridge_bank_bias_V[tgt_l].dtype))
                            if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_query_bank_module_replace":
                                if self._bridge_sample_query_features is None:
                                    raise ValueError(
                                        "bridge_ridge_qk_dynalign_value_query_bank_module_replace requires bridge sample query features; "
                                        "call set_bridge_sample_query_features() before fit_from_pairs"
                                    )
                                _, pred_fit_v = self._predict_bridge_query_module_replace(
                                    K_quant,
                                    K_pred,
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    module_slots,
                                    module_q,
                                    module_k,
                                    module_v,
                                    module_hidden,
                                    module_k_out,
                                    module_v_out,
                                )
                                query_bank = self._bridge_sample_query_features[tgt_l].to(device=Xk.device, dtype=torch.float32)
                                centroids, priors, sample_expert_ids = self._cluster_query_feature_bank(
                                    query_bank,
                                    num_clusters=self.config.bridge_bank_size,
                                )
                                self.bridge_bank_query_centroids[tgt_l].data.copy_(
                                    centroids.to(self.bridge_bank_query_centroids[tgt_l].dtype)
                                )
                                self.bridge_bank_priors[tgt_l].data.copy_(
                                    priors.to(self.bridge_bank_priors[tgt_l].dtype)
                                )
                                self.bridge_bank_templates[tgt_l].data.zero_()
                                sample_expert_ids = sample_expert_ids.to(device=Xk.device, dtype=torch.long)
                                min_samples = max(8, int(self.config.quantization_correction_rank or 8))
                                residual_target_v = Yv_fit - pred_fit_v
                                global_left_v, global_right_v, global_aux_left_v, global_aux_right_v = self._fit_bridge_query_residual_adapter(
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    pred_fit_v,
                                    residual_target_v,
                                    rank=int(self.config.quantization_correction_rank or 8),
                                )
                                zero_bias_k = torch.zeros_like(self.bridge_bank_bias_K[tgt_l].data[0])
                                zero_bias_v = torch.zeros_like(self.bridge_bank_bias_V[tgt_l].data[0])
                                for expert_idx in range(self.config.bridge_bank_size):
                                    mask = sample_expert_ids == expert_idx
                                    if int(mask.sum().item()) >= min_samples:
                                        left_v, right_v, aux_left_v, aux_right_v = self._fit_bridge_query_residual_adapter(
                                            V_quant[mask],
                                            V_pred[mask],
                                            query_features[mask],
                                            pred_fit_v[mask],
                                            residual_target_v[mask],
                                            rank=int(self.config.quantization_correction_rank or 8),
                                        )
                                    else:
                                        left_v, right_v = global_left_v, global_right_v
                                        aux_left_v, aux_right_v = global_aux_left_v, global_aux_right_v
                                    self.bridge_bank_proj_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_proj_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_aux_resid_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_aux_resid_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_bias_K[tgt_l].data[expert_idx].copy_(
                                        zero_bias_k.to(self.bridge_bank_bias_K[tgt_l].dtype)
                                    )
                                    self.bridge_bank_proj_V_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_proj_V_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_V_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_V_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_V_left[tgt_l].data[expert_idx].copy_(
                                        left_v.to(self.bridge_bank_query_resid_V_left[tgt_l].dtype)
                                    )
                                    self.bridge_bank_query_resid_V_right[tgt_l].data[expert_idx].copy_(
                                        right_v.to(self.bridge_bank_query_resid_V_right[tgt_l].dtype)
                                    )
                                    self.bridge_bank_query_aux_resid_V_left[tgt_l].data[expert_idx].copy_(
                                        aux_left_v.to(self.bridge_bank_query_aux_resid_V_left[tgt_l].dtype)
                                    )
                                    self.bridge_bank_query_aux_resid_V_right[tgt_l].data[expert_idx].copy_(
                                        aux_right_v.to(self.bridge_bank_query_aux_resid_V_right[tgt_l].dtype)
                                    )
                                    self.bridge_bank_bias_V[tgt_l].data[expert_idx].copy_(
                                        zero_bias_v.to(self.bridge_bank_bias_V[tgt_l].dtype)
                                    )
                            if self.config.quantization_correction == "bridge_ridge_qk_dynalign_value_bank_module_replace":
                                if self._bridge_prompt_cluster_labels is None or self._bridge_sample_prompt_ids is None:
                                    raise ValueError(
                                        "bridge_ridge_qk_dynalign_value_bank_module_replace requires bridge template-bank metadata; "
                                        "call set_bridge_runtime_template_bank() before fit_from_pairs"
                                    )
                                _, pred_fit_v = self._predict_bridge_query_module_replace(
                                    K_quant,
                                    K_pred,
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    module_slots,
                                    module_q,
                                    module_k,
                                    module_v,
                                    module_hidden,
                                    module_k_out,
                                    module_v_out,
                                )
                                prompt_ids = self._bridge_sample_prompt_ids.to(device=Xk.device)
                                cluster_labels = self._bridge_prompt_cluster_labels[tgt_l].to(device=Xk.device, dtype=torch.long)
                                sample_expert_ids = cluster_labels[prompt_ids]
                                min_samples = max(8, int(self.config.quantization_correction_rank or 8))
                                residual_target_v = Yv_fit - pred_fit_v
                                global_left_v, global_right_v, global_aux_left_v, global_aux_right_v = self._fit_bridge_query_residual_adapter(
                                    V_quant,
                                    V_pred,
                                    query_features,
                                    pred_fit_v,
                                    residual_target_v,
                                    rank=int(self.config.quantization_correction_rank or 8),
                                )
                                zero_bias_k = torch.zeros_like(self.bridge_bank_bias_K[tgt_l].data[0])
                                zero_bias_v = torch.zeros_like(self.bridge_bank_bias_V[tgt_l].data[0])
                                for expert_idx in range(self.config.bridge_bank_size):
                                    mask = sample_expert_ids == expert_idx
                                    if int(mask.sum().item()) >= min_samples:
                                        left_v, right_v, aux_left_v, aux_right_v = self._fit_bridge_query_residual_adapter(
                                            V_quant[mask],
                                            V_pred[mask],
                                            query_features[mask],
                                            pred_fit_v[mask],
                                            residual_target_v[mask],
                                            rank=int(self.config.quantization_correction_rank or 8),
                                        )
                                    else:
                                        left_v, right_v = global_left_v, global_right_v
                                        aux_left_v, aux_right_v = global_aux_left_v, global_aux_right_v
                                    self.bridge_bank_proj_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_proj_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_aux_resid_K_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_aux_resid_K_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_bias_K[tgt_l].data[expert_idx].copy_(
                                        zero_bias_k.to(self.bridge_bank_bias_K[tgt_l].dtype)
                                    )
                                    self.bridge_bank_proj_V_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_proj_V_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_V_left[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_aux_proj_V_right[tgt_l].data[expert_idx].zero_()
                                    self.bridge_bank_query_resid_V_left[tgt_l].data[expert_idx].copy_(
                                        left_v.to(self.bridge_bank_query_resid_V_left[tgt_l].dtype)
                                    )
                                    self.bridge_bank_query_resid_V_right[tgt_l].data[expert_idx].copy_(
                                        right_v.to(self.bridge_bank_query_resid_V_right[tgt_l].dtype)
                                    )
                                    self.bridge_bank_query_aux_resid_V_left[tgt_l].data[expert_idx].copy_(
                                        aux_left_v.to(self.bridge_bank_query_aux_resid_V_left[tgt_l].dtype)
                                    )
                                    self.bridge_bank_query_aux_resid_V_right[tgt_l].data[expert_idx].copy_(
                                        aux_right_v.to(self.bridge_bank_query_aux_resid_V_right[tgt_l].dtype)
                                    )
                                    self.bridge_bank_bias_V[tgt_l].data[expert_idx].copy_(
                                        zero_bias_v.to(self.bridge_bank_bias_V[tgt_l].dtype)
                                    )
                        if self.config.quantization_correction == "bridge_ridge_qk_tokenbasis_replace":
                            (
                                module_slots,
                                module_q,
                                module_k,
                                module_v,
                                module_hidden,
                                token_basis,
                                token_k_coeff,
                                token_v_coeff,
                            ) = self._fit_bridge_query_tokenbasis_replace(
                                K_quant,
                                K_pred,
                                V_quant,
                                V_pred,
                                query_features,
                                Yk_fit,
                                Yv_fit,
                                rank=int(self.config.quantization_correction_rank or 8),
                                prediction_distill_weight=0.25,
                                teacher_topk_log_probs=teacher_log_probs,
                                teacher_topk_output_rows=teacher_output_rows,
                            )
                            self.quant_query_module_slots[tgt_l].data.copy_(module_slots.to(self.quant_query_module_slots[tgt_l].dtype))
                            self.quant_query_module_q[tgt_l].data.copy_(module_q.to(self.quant_query_module_q[tgt_l].dtype))
                            self.quant_query_module_k[tgt_l].data.copy_(module_k.to(self.quant_query_module_k[tgt_l].dtype))
                            self.quant_query_module_v[tgt_l].data.copy_(module_v.to(self.quant_query_module_v[tgt_l].dtype))
                            self.quant_query_module_hidden[tgt_l].data.copy_(module_hidden.to(self.quant_query_module_hidden[tgt_l].dtype))
                            self.quant_query_token_basis[tgt_l].data.copy_(token_basis.to(self.quant_query_token_basis[tgt_l].dtype))
                            self.quant_query_token_K_coeff[tgt_l].data.copy_(token_k_coeff.to(self.quant_query_token_K_coeff[tgt_l].dtype))
                            self.quant_query_token_V_coeff[tgt_l].data.copy_(token_v_coeff.to(self.quant_query_token_V_coeff[tgt_l].dtype))
                        if self.config.quantization_correction in {"bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter"}:
                            (
                                shared_left,
                                shared_aux_left,
                                shared_k_right,
                                shared_v_right,
                                left_k,
                                right_k,
                                aux_left_k,
                                aux_right_k,
                                left_v,
                                right_v,
                                aux_left_v,
                                aux_right_v,
                            ) = self._fit_bridge_query_shared_residual_adapter(
                                K_quant,
                                K_pred,
                                V_quant,
                                V_pred,
                                query_features,
                                base_k,
                                base_v,
                                resid_target_k,
                                resid_target_v,
                                rank=int(self.config.quantization_correction_rank or 8),
                                prediction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_asym_predkl_adapter" else 0.0,
                                dynamic_prediction_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_asym_dynmap_adapter" else 0.0,
                                teacher_topk_log_probs=teacher_log_probs,
                                teacher_topk_output_rows=teacher_output_rows,
                            )
                            self.quant_query_shared_left[tgt_l].data.copy_(shared_left.to(self.quant_query_shared_left[tgt_l].dtype))
                            self.quant_query_shared_aux_left[tgt_l].data.copy_(shared_aux_left.to(self.quant_query_shared_aux_left[tgt_l].dtype))
                            self.quant_query_shared_K_right[tgt_l].data.copy_(shared_k_right.to(self.quant_query_shared_K_right[tgt_l].dtype))
                            self.quant_query_shared_V_right[tgt_l].data.copy_(shared_v_right.to(self.quant_query_shared_V_right[tgt_l].dtype))
                        elif self.config.quantization_correction == "bridge_ridge_qk_sae_adapter":
                            sparse_left, sparse_aux_left, sparse_k_right, sparse_v_right = self._fit_bridge_query_sparse_adapter(
                                K_quant,
                                K_pred,
                                V_quant,
                                V_pred,
                                query_features,
                                base_k,
                                base_v,
                                resid_target_k,
                                resid_target_v,
                                rank=int(self.config.quantization_correction_rank or 8),
                            )
                            self.quant_query_sparse_left[tgt_l].data.copy_(sparse_left.to(self.quant_query_sparse_left[tgt_l].dtype))
                            self.quant_query_sparse_aux_left[tgt_l].data.copy_(sparse_aux_left.to(self.quant_query_sparse_aux_left[tgt_l].dtype))
                            self.quant_query_sparse_K_right[tgt_l].data.copy_(sparse_k_right.to(self.quant_query_sparse_K_right[tgt_l].dtype))
                            self.quant_query_sparse_V_right[tgt_l].data.copy_(sparse_v_right.to(self.quant_query_sparse_V_right[tgt_l].dtype))
                        elif self.config.quantization_correction == "bridge_ridge_qk_generated_adapter":
                            (
                                hyper_left,
                                hyper_aux_left,
                                left_k,
                                right_k,
                                aux_left_k,
                                aux_right_k,
                                left_v,
                                right_v,
                                aux_left_v,
                                aux_right_v,
                            ) = self._fit_bridge_query_generated_adapter(
                                K_quant,
                                K_pred,
                                V_quant,
                                V_pred,
                                query_features,
                                base_k,
                                base_v,
                                resid_target_k,
                                resid_target_v,
                                rank=int(self.config.quantization_correction_rank or 8),
                                experts=int(self.config.bridge_bank_size),
                            )
                            self.quant_query_hyper_left[tgt_l].data.copy_(hyper_left.to(self.quant_query_hyper_left[tgt_l].dtype))
                            self.quant_query_hyper_aux_left[tgt_l].data.copy_(hyper_aux_left.to(self.quant_query_hyper_aux_left[tgt_l].dtype))
                            self.bridge_bank_query_resid_K_left[tgt_l].data.copy_(left_k.to(self.bridge_bank_query_resid_K_left[tgt_l].dtype))
                            self.bridge_bank_query_resid_K_right[tgt_l].data.copy_(right_k.to(self.bridge_bank_query_resid_K_right[tgt_l].dtype))
                            self.bridge_bank_query_aux_resid_K_left[tgt_l].data.copy_(aux_left_k.to(self.bridge_bank_query_aux_resid_K_left[tgt_l].dtype))
                            self.bridge_bank_query_aux_resid_K_right[tgt_l].data.copy_(aux_right_k.to(self.bridge_bank_query_aux_resid_K_right[tgt_l].dtype))
                            self.bridge_bank_query_resid_V_left[tgt_l].data.copy_(left_v.to(self.bridge_bank_query_resid_V_left[tgt_l].dtype))
                            self.bridge_bank_query_resid_V_right[tgt_l].data.copy_(right_v.to(self.bridge_bank_query_resid_V_right[tgt_l].dtype))
                            self.bridge_bank_query_aux_resid_V_left[tgt_l].data.copy_(aux_left_v.to(self.bridge_bank_query_aux_resid_V_left[tgt_l].dtype))
                            self.bridge_bank_query_aux_resid_V_right[tgt_l].data.copy_(aux_right_v.to(self.bridge_bank_query_aux_resid_V_right[tgt_l].dtype))
                        elif self.config.quantization_correction in {"bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace"}:
                            left_k = torch.zeros_like(self.quant_query_resid_K_left[tgt_l].data)
                            right_k = torch.zeros_like(self.quant_query_resid_K_right[tgt_l].data)
                            aux_left_k = torch.zeros_like(self.quant_query_aux_resid_K_left[tgt_l].data)
                            aux_right_k = torch.zeros_like(self.quant_query_aux_resid_K_right[tgt_l].data)
                            left_v = torch.zeros_like(self.quant_query_resid_V_left[tgt_l].data)
                            right_v = torch.zeros_like(self.quant_query_resid_V_right[tgt_l].data)
                            aux_left_v = torch.zeros_like(self.quant_query_aux_resid_V_left[tgt_l].data)
                            aux_right_v = torch.zeros_like(self.quant_query_aux_resid_V_right[tgt_l].data)
                        else:
                            left_k, right_k, aux_left_k, aux_right_k = self._fit_bridge_query_residual_adapter(
                                K_quant,
                                K_pred,
                                query_features,
                                base_k,
                                resid_target_k,
                                rank=int(self.config.quantization_correction_rank or 8),
                                affinity_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_affinity_adapter" else 0.0,
                                attention_kl_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_attnkl_adapter" else 0.0,
                                local_attention_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_cab_adapter" else 0.0,
                                interaction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_emkd_adapter" else 0.0,
                                readout_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_readout_adapter" else 0.0,
                                prediction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_predkl_adapter" else 0.0,
                                teacher_topk_log_probs=teacher_log_probs,
                                teacher_topk_output_rows=teacher_output_rows,
                                readout_partner=Yv_fit if self.config.quantization_correction == "bridge_ridge_qk_readout_adapter" else None,
                                readout_partner_kind="V" if self.config.quantization_correction == "bridge_ridge_qk_readout_adapter" else None,
                                sample_prompt_ids=sample_prompt_ids,
                            )
                            left_v, right_v, aux_left_v, aux_right_v = self._fit_bridge_query_residual_adapter(
                                V_quant,
                                V_pred,
                                query_features,
                                base_v,
                                resid_target_v,
                                rank=int(self.config.quantization_correction_rank or 8),
                                affinity_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_affinity_adapter" else 0.0,
                                attention_kl_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_attnkl_adapter" else 0.0,
                                local_attention_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_cab_adapter" else 0.0,
                                interaction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_emkd_adapter" else 0.0,
                                readout_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_readout_adapter" else 0.0,
                                prediction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_predkl_adapter" else 0.0,
                                teacher_topk_log_probs=teacher_log_probs,
                                teacher_topk_output_rows=teacher_output_rows,
                                readout_partner=Yk_fit if self.config.quantization_correction == "bridge_ridge_qk_readout_adapter" else None,
                                readout_partner_kind="K" if self.config.quantization_correction == "bridge_ridge_qk_readout_adapter" else None,
                                sample_prompt_ids=sample_prompt_ids,
                            )
                        if self.config.quantization_correction not in {"bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace"}:
                            self.quant_query_resid_K_left[tgt_l].data.copy_(left_k.to(self.quant_query_resid_K_left[tgt_l].dtype))
                            self.quant_query_resid_K_right[tgt_l].data.copy_(right_k.to(self.quant_query_resid_K_right[tgt_l].dtype))
                            self.quant_query_aux_resid_K_left[tgt_l].data.copy_(aux_left_k.to(self.quant_query_aux_resid_K_left[tgt_l].dtype))
                            self.quant_query_aux_resid_K_right[tgt_l].data.copy_(aux_right_k.to(self.quant_query_aux_resid_K_right[tgt_l].dtype))
                            self.quant_query_resid_V_left[tgt_l].data.copy_(left_v.to(self.quant_query_resid_V_left[tgt_l].dtype))
                            self.quant_query_resid_V_right[tgt_l].data.copy_(right_v.to(self.quant_query_resid_V_right[tgt_l].dtype))
                            self.quant_query_aux_resid_V_left[tgt_l].data.copy_(aux_left_v.to(self.quant_query_aux_resid_V_left[tgt_l].dtype))
                            self.quant_query_aux_resid_V_right[tgt_l].data.copy_(aux_right_v.to(self.quant_query_aux_resid_V_right[tgt_l].dtype))
                self.quant_proj_K[tgt_l].data.copy_(proj_k.to(self.quant_proj_K[tgt_l].dtype))
                self.quant_proj_V[tgt_l].data.copy_(proj_v.to(self.quant_proj_V[tgt_l].dtype))
                self.quant_aux_proj_K[tgt_l].data.copy_(aux_proj_k.to(self.quant_aux_proj_K[tgt_l].dtype))
                self.quant_aux_proj_V[tgt_l].data.copy_(aux_proj_v.to(self.quant_aux_proj_V[tgt_l].dtype))
                self.quant_bias_K[tgt_l].data.copy_(bias_k.to(self.quant_bias_K[tgt_l].dtype))
                self.quant_bias_V[tgt_l].data.copy_(bias_v.to(self.quant_bias_V[tgt_l].dtype))
                if self.config.quantization_correction in {"bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                    if self._bridge_prompt_cluster_labels is None or self._bridge_sample_prompt_ids is None:
                        raise ValueError(
                            "bridge_ridge_residual_bank requires bridge template-bank metadata; "
                            "call set_bridge_runtime_template_bank() before fit_from_pairs"
                        )
                    prompt_ids = self._bridge_sample_prompt_ids.to(device=Xk.device)
                    cluster_labels = self._bridge_prompt_cluster_labels[tgt_l].to(device=Xk.device, dtype=torch.long)
                    sample_expert_ids = cluster_labels[prompt_ids]
                    rank = self.config.quantization_correction_rank
                    min_samples = max(8, int(rank or 8))
                    base_k = K_quant @ proj_k + K_pred @ aux_proj_k + bias_k
                    base_v = V_quant @ proj_v + V_pred @ aux_proj_v + bias_v
                    residual_target_k = Yk_fit - base_k
                    residual_target_v = Yv_fit - base_v
                    if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                        if self._bridge_sample_query_features is None:
                            raise ValueError(
                                f"{self.config.quantization_correction} requires bridge sample query features; "
                                "call set_bridge_sample_query_features() before fit_from_pairs"
                            )
                        query_features = self._bridge_sample_query_features[tgt_l].to(device=Xk.device)
                        teacher_log_probs = None
                        teacher_output_rows = None
                        if self.config.quantization_correction == "bridge_ridge_qk_predkl_bank":
                            if self._bridge_prediction_teacher_log_probs is None or self._bridge_prediction_teacher_output_rows is None:
                                raise ValueError(
                                    "bridge_ridge_qk_predkl_bank requires prediction-teacher tensors; "
                                    "call set_bridge_prediction_teacher() before fit_from_pairs"
                                )
                            teacher_log_probs = self._bridge_prediction_teacher_log_probs.to(device=Xk.device)
                            teacher_output_rows = self._bridge_prediction_teacher_output_rows.to(device=Xk.device)
                        global_left_k, global_right_k, global_aux_left_k, global_aux_right_k = self._fit_bridge_query_residual_adapter(
                            K_quant,
                            K_pred,
                            query_features,
                            base_k,
                            residual_target_k,
                            rank=int(self.config.quantization_correction_rank or 8),
                            local_attention_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" else 0.0,
                            prediction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_predkl_bank" else 0.0,
                            teacher_topk_log_probs=teacher_log_probs,
                            teacher_topk_output_rows=teacher_output_rows,
                            sample_prompt_ids=prompt_ids if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" else None,
                        )
                        global_left_v, global_right_v, global_aux_left_v, global_aux_right_v = self._fit_bridge_query_residual_adapter(
                            V_quant,
                            V_pred,
                            query_features,
                            base_v,
                            residual_target_v,
                            rank=int(self.config.quantization_correction_rank or 8),
                            local_attention_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" else 0.0,
                            prediction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_predkl_bank" else 0.0,
                            teacher_topk_log_probs=teacher_log_probs,
                            teacher_topk_output_rows=teacher_output_rows,
                            sample_prompt_ids=prompt_ids if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" else None,
                        )
                        global_resid_bias_k = torch.zeros_like(self.bridge_bank_bias_K[tgt_l].data[0])
                        global_resid_bias_v = torch.zeros_like(self.bridge_bank_bias_V[tgt_l].data[0])
                    else:
                        global_resid_proj_k, global_resid_aux_proj_k, global_resid_bias_k = self._fit_bridge_ridge_correction(
                            K_quant,
                            K_pred,
                            residual_target_k,
                            lam=self.config.ridge_lambda,
                        )
                        global_resid_proj_v, global_resid_aux_proj_v, global_resid_bias_v = self._fit_bridge_ridge_correction(
                            V_quant,
                            V_pred,
                            residual_target_v,
                            lam=self.config.ridge_lambda,
                        )
                        global_left_k, global_right_k = self._factorize_low_rank_matrix(global_resid_proj_k, rank=rank)
                        global_aux_left_k, global_aux_right_k = self._factorize_low_rank_matrix(global_resid_aux_proj_k, rank=rank)
                        global_left_v, global_right_v = self._factorize_low_rank_matrix(global_resid_proj_v, rank=rank)
                        global_aux_left_v, global_aux_right_v = self._factorize_low_rank_matrix(global_resid_aux_proj_v, rank=rank)
                    for expert_idx in range(self.config.bridge_bank_size):
                        mask = sample_expert_ids == expert_idx
                        if int(mask.sum().item()) >= min_samples:
                            if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                                prompt_ids_mask = prompt_ids[mask]
                                left_k, right_k, aux_left_k, aux_right_k = self._fit_bridge_query_residual_adapter(
                                    K_quant[mask],
                                    K_pred[mask],
                                    query_features[mask],
                                    base_k[mask],
                                    residual_target_k[mask],
                                    rank=int(self.config.quantization_correction_rank or 8),
                                    local_attention_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" else 0.0,
                                    prediction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_predkl_bank" else 0.0,
                                    teacher_topk_log_probs=teacher_log_probs[mask] if teacher_log_probs is not None else None,
                                    teacher_topk_output_rows=teacher_output_rows[mask] if teacher_output_rows is not None else None,
                                    sample_prompt_ids=prompt_ids_mask if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" else None,
                                )
                                left_v, right_v, aux_left_v, aux_right_v = self._fit_bridge_query_residual_adapter(
                                    V_quant[mask],
                                    V_pred[mask],
                                    query_features[mask],
                                    base_v[mask],
                                    residual_target_v[mask],
                                    rank=int(self.config.quantization_correction_rank or 8),
                                    local_attention_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" else 0.0,
                                    prediction_distill_weight=0.25 if self.config.quantization_correction == "bridge_ridge_qk_predkl_bank" else 0.0,
                                    teacher_topk_log_probs=teacher_log_probs[mask] if teacher_log_probs is not None else None,
                                    teacher_topk_output_rows=teacher_output_rows[mask] if teacher_output_rows is not None else None,
                                    sample_prompt_ids=prompt_ids_mask if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" else None,
                                )
                                bias_resid_k = global_resid_bias_k
                                bias_resid_v = global_resid_bias_v
                            else:
                                resid_proj_k, resid_aux_proj_k, resid_bias_k = self._fit_bridge_ridge_correction(
                                    K_quant[mask],
                                    K_pred[mask],
                                    residual_target_k[mask],
                                    lam=self.config.ridge_lambda,
                                )
                                resid_proj_v, resid_aux_proj_v, resid_bias_v = self._fit_bridge_ridge_correction(
                                    V_quant[mask],
                                    V_pred[mask],
                                    residual_target_v[mask],
                                    lam=self.config.ridge_lambda,
                                )
                                left_k, right_k = self._factorize_low_rank_matrix(resid_proj_k, rank=rank)
                                aux_left_k, aux_right_k = self._factorize_low_rank_matrix(resid_aux_proj_k, rank=rank)
                                left_v, right_v = self._factorize_low_rank_matrix(resid_proj_v, rank=rank)
                                aux_left_v, aux_right_v = self._factorize_low_rank_matrix(resid_aux_proj_v, rank=rank)
                                bias_resid_k = resid_bias_k
                                bias_resid_v = resid_bias_v
                        else:
                            left_k, right_k = global_left_k, global_right_k
                            aux_left_k, aux_right_k = global_aux_left_k, global_aux_right_k
                            bias_resid_k = global_resid_bias_k
                            left_v, right_v = global_left_v, global_right_v
                            aux_left_v, aux_right_v = global_aux_left_v, global_aux_right_v
                            bias_resid_v = global_resid_bias_v
                        if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                            self.bridge_bank_proj_K_left[tgt_l].data[expert_idx].zero_()
                            self.bridge_bank_proj_K_right[tgt_l].data[expert_idx].zero_()
                            self.bridge_bank_aux_proj_K_left[tgt_l].data[expert_idx].zero_()
                            self.bridge_bank_aux_proj_K_right[tgt_l].data[expert_idx].zero_()
                        else:
                            self.bridge_bank_proj_K_left[tgt_l].data[expert_idx].copy_(
                                left_k.to(self.bridge_bank_proj_K_left[tgt_l].dtype)
                            )
                            self.bridge_bank_proj_K_right[tgt_l].data[expert_idx].copy_(
                                right_k.to(self.bridge_bank_proj_K_right[tgt_l].dtype)
                            )
                            self.bridge_bank_aux_proj_K_left[tgt_l].data[expert_idx].copy_(
                                aux_left_k.to(self.bridge_bank_aux_proj_K_left[tgt_l].dtype)
                            )
                            self.bridge_bank_aux_proj_K_right[tgt_l].data[expert_idx].copy_(
                                aux_right_k.to(self.bridge_bank_aux_proj_K_right[tgt_l].dtype)
                            )
                        if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                            self.bridge_bank_query_resid_K_left[tgt_l].data[expert_idx].copy_(
                                left_k.to(self.bridge_bank_query_resid_K_left[tgt_l].dtype)
                            )
                            self.bridge_bank_query_resid_K_right[tgt_l].data[expert_idx].copy_(
                                right_k.to(self.bridge_bank_query_resid_K_right[tgt_l].dtype)
                            )
                            self.bridge_bank_query_aux_resid_K_left[tgt_l].data[expert_idx].copy_(
                                aux_left_k.to(self.bridge_bank_query_aux_resid_K_left[tgt_l].dtype)
                            )
                            self.bridge_bank_query_aux_resid_K_right[tgt_l].data[expert_idx].copy_(
                                aux_right_k.to(self.bridge_bank_query_aux_resid_K_right[tgt_l].dtype)
                            )
                        self.bridge_bank_bias_K[tgt_l].data[expert_idx].copy_(
                            bias_resid_k.to(self.bridge_bank_bias_K[tgt_l].dtype)
                        )
                        if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                            self.bridge_bank_proj_V_left[tgt_l].data[expert_idx].zero_()
                            self.bridge_bank_proj_V_right[tgt_l].data[expert_idx].zero_()
                            self.bridge_bank_aux_proj_V_left[tgt_l].data[expert_idx].zero_()
                            self.bridge_bank_aux_proj_V_right[tgt_l].data[expert_idx].zero_()
                        else:
                            self.bridge_bank_proj_V_left[tgt_l].data[expert_idx].copy_(
                                left_v.to(self.bridge_bank_proj_V_left[tgt_l].dtype)
                            )
                            self.bridge_bank_proj_V_right[tgt_l].data[expert_idx].copy_(
                                right_v.to(self.bridge_bank_proj_V_right[tgt_l].dtype)
                            )
                            self.bridge_bank_aux_proj_V_left[tgt_l].data[expert_idx].copy_(
                                aux_left_v.to(self.bridge_bank_aux_proj_V_left[tgt_l].dtype)
                            )
                            self.bridge_bank_aux_proj_V_right[tgt_l].data[expert_idx].copy_(
                                aux_right_v.to(self.bridge_bank_aux_proj_V_right[tgt_l].dtype)
                            )
                        if self.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                            self.bridge_bank_query_resid_V_left[tgt_l].data[expert_idx].copy_(
                                left_v.to(self.bridge_bank_query_resid_V_left[tgt_l].dtype)
                            )
                            self.bridge_bank_query_resid_V_right[tgt_l].data[expert_idx].copy_(
                                right_v.to(self.bridge_bank_query_resid_V_right[tgt_l].dtype)
                            )
                            self.bridge_bank_query_aux_resid_V_left[tgt_l].data[expert_idx].copy_(
                                aux_left_v.to(self.bridge_bank_query_aux_resid_V_left[tgt_l].dtype)
                            )
                            self.bridge_bank_query_aux_resid_V_right[tgt_l].data[expert_idx].copy_(
                                aux_right_v.to(self.bridge_bank_query_aux_resid_V_right[tgt_l].dtype)
                            )
                        self.bridge_bank_bias_V[tgt_l].data[expert_idx].copy_(
                            bias_resid_v.to(self.bridge_bank_bias_V[tgt_l].dtype)
                        )
            elif self.config.quantization_correction == "bridge_low_rank_bank":
                if self._bridge_prompt_cluster_labels is None or self._bridge_sample_prompt_ids is None:
                    raise ValueError(
                        "bridge_low_rank_bank requires bridge template-bank metadata; "
                        "call set_bridge_runtime_template_bank() before fit_from_pairs"
                    )
                prompt_ids = self._bridge_sample_prompt_ids.to(device=Xk.device)
                cluster_labels = self._bridge_prompt_cluster_labels[tgt_l].to(device=Xk.device, dtype=torch.long)
                sample_expert_ids = cluster_labels[prompt_ids]
                rank = self.config.quantization_correction_rank
                global_proj_k, global_aux_proj_k, global_bias_k = self._fit_bridge_ridge_correction(
                    K_quant,
                    K_pred,
                    Yk_fit,
                    lam=self.config.ridge_lambda,
                )
                global_proj_v, global_aux_proj_v, global_bias_v = self._fit_bridge_ridge_correction(
                    V_quant,
                    V_pred,
                    Yv_fit,
                    lam=self.config.ridge_lambda,
                )
                global_left_k, global_right_k = self._factorize_low_rank_matrix(global_proj_k, rank=rank)
                global_aux_left_k, global_aux_right_k = self._factorize_low_rank_matrix(global_aux_proj_k, rank=rank)
                global_left_v, global_right_v = self._factorize_low_rank_matrix(global_proj_v, rank=rank)
                global_aux_left_v, global_aux_right_v = self._factorize_low_rank_matrix(global_aux_proj_v, rank=rank)
                min_samples = max(8, int(rank or 8))
                for expert_idx in range(self.config.bridge_bank_size):
                    mask = sample_expert_ids == expert_idx
                    if int(mask.sum().item()) >= min_samples:
                        proj_k, aux_proj_k, bias_k = self._fit_bridge_ridge_correction(
                            K_quant[mask],
                            K_pred[mask],
                            Yk_fit[mask],
                            lam=self.config.ridge_lambda,
                        )
                        proj_v, aux_proj_v, bias_v = self._fit_bridge_ridge_correction(
                            V_quant[mask],
                            V_pred[mask],
                            Yv_fit[mask],
                            lam=self.config.ridge_lambda,
                        )
                        left_k, right_k = self._factorize_low_rank_matrix(proj_k, rank=rank)
                        aux_left_k, aux_right_k = self._factorize_low_rank_matrix(aux_proj_k, rank=rank)
                        left_v, right_v = self._factorize_low_rank_matrix(proj_v, rank=rank)
                        aux_left_v, aux_right_v = self._factorize_low_rank_matrix(aux_proj_v, rank=rank)
                    else:
                        left_k, right_k = global_left_k, global_right_k
                        aux_left_k, aux_right_k = global_aux_left_k, global_aux_right_k
                        bias_k = global_bias_k
                        left_v, right_v = global_left_v, global_right_v
                        aux_left_v, aux_right_v = global_aux_left_v, global_aux_right_v
                        bias_v = global_bias_v
                    self.bridge_bank_proj_K_left[tgt_l].data[expert_idx].copy_(
                        left_k.to(self.bridge_bank_proj_K_left[tgt_l].dtype)
                    )
                    self.bridge_bank_proj_K_right[tgt_l].data[expert_idx].copy_(
                        right_k.to(self.bridge_bank_proj_K_right[tgt_l].dtype)
                    )
                    self.bridge_bank_aux_proj_K_left[tgt_l].data[expert_idx].copy_(
                        aux_left_k.to(self.bridge_bank_aux_proj_K_left[tgt_l].dtype)
                    )
                    self.bridge_bank_aux_proj_K_right[tgt_l].data[expert_idx].copy_(
                        aux_right_k.to(self.bridge_bank_aux_proj_K_right[tgt_l].dtype)
                    )
                    self.bridge_bank_bias_K[tgt_l].data[expert_idx].copy_(
                        bias_k.to(self.bridge_bank_bias_K[tgt_l].dtype)
                    )
                    self.bridge_bank_proj_V_left[tgt_l].data[expert_idx].copy_(
                        left_v.to(self.bridge_bank_proj_V_left[tgt_l].dtype)
                    )
                    self.bridge_bank_proj_V_right[tgt_l].data[expert_idx].copy_(
                        right_v.to(self.bridge_bank_proj_V_right[tgt_l].dtype)
                    )
                    self.bridge_bank_aux_proj_V_left[tgt_l].data[expert_idx].copy_(
                        aux_left_v.to(self.bridge_bank_aux_proj_V_left[tgt_l].dtype)
                    )
                    self.bridge_bank_aux_proj_V_right[tgt_l].data[expert_idx].copy_(
                        aux_right_v.to(self.bridge_bank_aux_proj_V_right[tgt_l].dtype)
                    )
                    self.bridge_bank_bias_V[tgt_l].data[expert_idx].copy_(
                        bias_v.to(self.bridge_bank_bias_V[tgt_l].dtype)
                    )
            elif self.config.quantization_correction == "ridge":
                proj_k, bias_k = self._fit_ridge_correction(K_quant, Yk_fit, lam=self.config.ridge_lambda)
                proj_v, bias_v = self._fit_ridge_correction(V_quant, Yv_fit, lam=self.config.ridge_lambda)
                self.quant_proj_K[tgt_l].data.copy_(proj_k.to(self.quant_proj_K[tgt_l].dtype))
                self.quant_proj_V[tgt_l].data.copy_(proj_v.to(self.quant_proj_V[tgt_l].dtype))
                self.quant_bias_K[tgt_l].data.copy_(bias_k.to(self.quant_bias_K[tgt_l].dtype))
                self.quant_bias_V[tgt_l].data.copy_(bias_v.to(self.quant_bias_V[tgt_l].dtype))
            elif self.config.quantization_correction == "low_rank":
                proj_k, bias_k = self._fit_low_rank_correction(
                    K_quant,
                    Yk_fit,
                    rank=self.config.quantization_correction_rank,
                    lam=self.config.ridge_lambda,
                )
                proj_v, bias_v = self._fit_low_rank_correction(
                    V_quant,
                    Yv_fit,
                    rank=self.config.quantization_correction_rank,
                    lam=self.config.ridge_lambda,
                )
                self.quant_proj_K[tgt_l].data.copy_(proj_k.to(self.quant_proj_K[tgt_l].dtype))
                self.quant_proj_V[tgt_l].data.copy_(proj_v.to(self.quant_proj_V[tgt_l].dtype))
                self.quant_bias_K[tgt_l].data.copy_(bias_k.to(self.quant_bias_K[tgt_l].dtype))
                self.quant_bias_V[tgt_l].data.copy_(bias_v.to(self.quant_bias_V[tgt_l].dtype))
            elif self.config.quantization_correction != "none":
                raise ValueError(f"Unknown quantization_correction: {self.config.quantization_correction}")

            fit_runtime_query_features = None
            if self.config.quantization_correction in {"bridge_ridge_qk_projector", "bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter", "bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                if self._bridge_sample_query_features is None:
                    raise ValueError(
                        f"{self.config.quantization_correction} requires bridge sample query features during fit; "
                        "call set_bridge_sample_query_features() before fit_from_pairs"
                    )
                fit_runtime_query_features = self._bridge_sample_query_features[tgt_l].to(device=Xk.device).view(
                    K_quant.shape[0], 1, self.d_t
                )
            K_runtime = K_quant if self.config.quantization_correction == "none" else self._apply_quantization_correction(
                K_quant,
                tgt_l,
                "K",
                aux_input=K_pred,
                paired_input=V_quant,
                paired_aux_input=V_pred,
                runtime_query_features=fit_runtime_query_features,
            )
            V_runtime = V_quant if self.config.quantization_correction == "none" else self._apply_quantization_correction(
                V_quant,
                tgt_l,
                "V",
                aux_input=V_pred,
                paired_input=K_quant,
                paired_aux_input=K_pred,
                runtime_query_features=fit_runtime_query_features,
            )
            if self._target_whitening_applies(tgt_l, "k"):
                K_runtime = undo_whitening(
                    K_runtime,
                    self.whiten_K_tgt_inv[tgt_l],
                    self.whiten_K_tgt_mean[tgt_l],
                )
            if self._target_whitening_applies(tgt_l, "v"):
                V_runtime = undo_whitening(
                    V_runtime,
                    self.whiten_V_tgt_inv[tgt_l],
                    self.whiten_V_tgt_mean[tgt_l],
                )
            B, S = Ks_rot.shape[0], Ks_rot.shape[1]
            K_runtime_orig = self._unflatten_and_inverse_rotate(
                K_runtime.view(B, S, self.d_t),
                self.config.tgt_num_heads,
                self.config.tgt_head_dim,
                self.R_t.T,
            )
            V_runtime_orig = self._unflatten_and_inverse_rotate(
                V_runtime.view(B, S, self.d_t),
                self.config.tgt_num_heads,
                self.config.tgt_head_dim,
                self.R_t.T,
            )
            Yk_orig = self._flatten_without_rotation(K_t).reshape(-1, self.d_t)
            Yv_orig = self._flatten_without_rotation(V_t).reshape(-1, self.d_t)
            K_runtime_orig_flat = self._flatten_without_rotation(K_runtime_orig).reshape(-1, self.d_t)
            V_runtime_orig_flat = self._flatten_without_rotation(V_runtime_orig).reshape(-1, self.d_t)
            src_scale_k, tgt_scale_k, bias_k = self._fit_coordinate_fuser(
                K_runtime_orig_flat,
                Yk_orig,
                dropout=float(self.config.learned_fusion_dropout),
                salt=tgt_l * 2,
            )
            src_scale_v, tgt_scale_v, bias_v = self._fit_coordinate_fuser(
                V_runtime_orig_flat,
                Yv_orig,
                dropout=float(self.config.learned_fusion_dropout),
                salt=tgt_l * 2 + 1,
            )
            self.fusion_src_scale_K[tgt_l].data.copy_(src_scale_k.to(self.fusion_src_scale_K[tgt_l].dtype))
            self.fusion_src_scale_V[tgt_l].data.copy_(src_scale_v.to(self.fusion_src_scale_V[tgt_l].dtype))
            self.fusion_tgt_scale_K[tgt_l].data.copy_(tgt_scale_k.to(self.fusion_tgt_scale_K[tgt_l].dtype))
            self.fusion_tgt_scale_V[tgt_l].data.copy_(tgt_scale_v.to(self.fusion_tgt_scale_V[tgt_l].dtype))
            self.fusion_bias_K[tgt_l].data.copy_(bias_k.to(self.fusion_bias_K[tgt_l].dtype))
            self.fusion_bias_V[tgt_l].data.copy_(bias_v.to(self.fusion_bias_V[tgt_l].dtype))
            if self.fusion_head_proj_K is not None and self.fusion_head_proj_V is not None:
                head_proj_k, head_bias_k = self._fit_head_ridge_fuser(
                    K_runtime_orig,
                    K_t,
                    dropout=float(self.config.learned_fusion_dropout),
                    salt=10_000 + tgt_l * 2,
                )
                head_proj_v, head_bias_v = self._fit_head_ridge_fuser(
                    V_runtime_orig,
                    V_t,
                    dropout=float(self.config.learned_fusion_dropout),
                    salt=10_000 + tgt_l * 2 + 1,
                )
                self.fusion_head_proj_K[tgt_l].data.copy_(head_proj_k.to(self.fusion_head_proj_K[tgt_l].dtype))
                self.fusion_head_proj_V[tgt_l].data.copy_(head_proj_v.to(self.fusion_head_proj_V[tgt_l].dtype))
                self.fusion_head_bias_K[tgt_l].data.copy_(head_bias_k.to(self.fusion_head_bias_K[tgt_l].dtype))
                self.fusion_head_bias_V[tgt_l].data.copy_(head_bias_v.to(self.fusion_head_bias_V[tgt_l].dtype))

            q_k = alignment_quality(Xk, Yk_fit, W_K)
            q_v = alignment_quality(Xv, Yv_fit, W_V)
            if self._target_whitening_applies(tgt_l, "k"):
                Yk_hat = Xk @ W_K
                q_k["original_space_relative_frobenius_error"] = float(
                    (undo_whitening(Yk_hat, self.whiten_K_tgt_inv[tgt_l], self.whiten_K_tgt_mean[tgt_l]) - Yk).norm()
                    / (Yk.norm() + 1e-12)
                )
            if self._target_whitening_applies(tgt_l, "v"):
                Yv_hat = Xv @ W_V
                q_v["original_space_relative_frobenius_error"] = float(
                    (undo_whitening(Yv_hat, self.whiten_V_tgt_inv[tgt_l], self.whiten_V_tgt_mean[tgt_l]) - Yv).norm()
                    / (Yv.norm() + 1e-12)
                )
            diagnostics[tgt_l] = {"K": q_k, "V": q_v, "src_layer": src_l}
            if grouped_alignment and self.config.alignment_method in {"grouped_transport", "grouped_permutation", "grouped_signature_transport", "grouped_subspace_transport", "grouped_canonical_transport", "grouped_adaptive_canonical_transport", "grouped_rotational_transport", "grouped_fitted_rotation_transport", "grouped_shared_basis_transport", "grouped_covariance_transport", "grouped_template_transport", "grouped_qk_retrieval_transport", "grouped_contrastive_template_transport", "grouped_template_subspace_transport"}:
                diagnostics[tgt_l]["K_transport_plan"] = self.transport_plan_K[tgt_l].detach().cpu().tolist()
                diagnostics[tgt_l]["V_transport_plan"] = self.transport_plan_V[tgt_l].detach().cpu().tolist()
            elif self.config.alignment_method in {
                "broadcast_template_transport",
                "broadcast_template_ot_transport",
                "broadcast_peak_template_ot_transport",
                "broadcast_retrieval_spectrum_ot_transport",
                "broadcast_qk_template_ot_transport",
            } and self._broadcast_transport_plan_K is not None and self._broadcast_transport_plan_V is not None:
                diagnostics[tgt_l]["K_transport_plan"] = self._broadcast_transport_plan_K[tgt_l].detach().cpu().tolist()
                diagnostics[tgt_l]["V_transport_plan"] = self._broadcast_transport_plan_V[tgt_l].detach().cpu().tolist()

            # Fit optional head-group saliency from local aligned slices.
            group_scores: list[tuple[float, int]] = []
            base_method = self.config.alignment_method.removeprefix("grouped_")
            if base_method in {
                "transport",
                "permutation",
                "signature_transport",
                "subspace_transport",
                "canonical_transport",
                "adaptive_canonical_transport",
                "rotational_transport",
                "fitted_rotation_transport",
                "shared_basis_transport",
                "covariance_transport",
                "template_transport",
                "qk_retrieval_transport",
                "contrastive_template_transport",
                "template_subspace_transport",
                "broadcast_template_transport",
                "broadcast_template_ot_transport",
                "broadcast_peak_template_ot_transport",
                "broadcast_retrieval_spectrum_ot_transport",
                "broadcast_qk_template_ot_transport",
            }:
                base_method = "auto"
            for group_idx, (src_slice, tgt_slice) in enumerate(
                zip(self._group_feature_slices(use_target=False), self._group_feature_slices(use_target=True))
            ):
                WgK = fit_alignment(
                    Xk[:, src_slice],
                    Yk_fit[:, tgt_slice],
                    method=base_method,
                    lam=lam_k,
                    rank=self.config.alignment_rank,
                )
                WgV = fit_alignment(
                    Xv[:, src_slice],
                    Yv_fit[:, tgt_slice],
                    method=base_method,
                    lam=lam_v,
                    rank=self.config.alignment_rank,
                )
                qg_k = alignment_quality(Xk[:, src_slice], Yk_fit[:, tgt_slice], WgK)
                qg_v = alignment_quality(Xv[:, src_slice], Yv_fit[:, tgt_slice], WgV)
                if self.config.head_selection_metric == "mean_cosine_similarity":
                    score = 0.5 * (
                        qg_k["mean_cosine_similarity"] + qg_v["mean_cosine_similarity"]
                    )
                elif self.config.head_selection_metric == "negative_error":
                    score = -0.5 * (
                        qg_k["relative_frobenius_error"] + qg_v["relative_frobenius_error"]
                    )
                else:
                    raise ValueError(
                        f"Unknown head_selection_metric: {self.config.head_selection_metric}"
                    )
                group_scores.append((float(score), group_idx))
            self._apply_head_selection_from_scores(group_scores, tgt_l)
            if verbose:
                print(
                    f"[layer {tgt_l:>2d} <- src {src_l:>2d}]  "
                    f"K rel_err={q_k['relative_frobenius_error']:.3f}  "
                    f"cos={q_k['mean_cosine_similarity']:.3f}  |  "
                    f"V rel_err={q_v['relative_frobenius_error']:.3f}  "
                    f"cos={q_v['mean_cosine_similarity']:.3f}"
                )

        self._apply_layer_selection(diagnostics)
        self._fitted = True
        return diagnostics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "RotAlignKVTranslator":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = TranslatorConfig(**payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"], strict=False)
        model._fitted = payload.get("fitted", True)
        return model
