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

    # Alignment solver: 'auto' | 'identity' | 'procrustes'
    #                 | 'procrustes_rand' | 'ridge' | 'cca' | 'reduced_rank'
    #                 | 'grouped_' + any of the above
    #                 | 'grouped_transport' | 'grouped_permutation'
    #                 | 'grouped_signature_transport' | 'grouped_subspace_transport'
    #                 | 'grouped_canonical_transport' | 'grouped_covariance_transport'
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
    # adapter trained with prompt-local causal attention behavior, and
    # `bridge_ridge_qk_weighted` keeps the global bridge
    # but fits it with calibration samples reweighted by target retrieval
    # importance, `bridge_ridge_qk_projector` feeds live query-conditioned
    # projector features directly into the bridge itself, `bridge_ridge_qk_adapter`
    # adds a tiny learned low-rank query-conditioned residual on top of the
    # global bridge, `bridge_ridge_qk_affinity_adapter` adds a
    # query-conditioned affinity-matching objective to that same residual fit,
    # `bridge_ridge_qk_attnkl_adapter` adds sampled attention-logit KL on top
    # of the same residual fit, `bridge_ridge_qk_cab_adapter` swaps that
    # local teacher target for prompt-local causal attention-behavior
    # distillation inspired by CAB, and `bridge_ridge_qk_emkd_adapter`
    # instead matches prompt-local token-interaction distributions inspired by
    # richer relational distillation; `ridge` learns a full linear map plus bias
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
            W_resid = fit_alignment(
                X,
                Y - base_pred,
                method="reduced_rank",
                lam=lam,
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
        if float(local_attention_weight) > 0.0 or float(interaction_distill_weight) > 0.0:
            if sample_prompt_ids is None:
                raise ValueError(
                    "sample_prompt_ids are required when local_attention_weight > 0 or interaction_distill_weight > 0"
                )
            prompt_ids_cpu = sample_prompt_ids.detach().to("cpu", dtype=torch.long).view(-1)
            if prompt_ids_cpu.numel() != q.shape[0]:
                raise ValueError(
                    "sample_prompt_ids must align with calibration samples, "
                    f"got {int(prompt_ids_cpu.numel())} ids for {q.shape[0]} samples"
                )
            unique_prompt_ids = torch.unique(prompt_ids_cpu)

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
                loss.backward()
                optimizer.step()

        return (
            q_left.detach().to(dtype=residual_target.dtype, device=residual_target.device),
            q_right.detach().to(dtype=residual_target.dtype, device=residual_target.device),
            aux_left.detach().to(dtype=residual_target.dtype, device=residual_target.device),
            aux_right.detach().to(dtype=residual_target.dtype, device=residual_target.device),
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

    def _apply_quantization_correction(
        self,
        x: torch.Tensor,
        tgt_layer_idx: int,
        kind: str,
        *,
        aux_input: torch.Tensor | None = None,
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
        if self.config.quantization_correction in {"bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter"}:
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
            resid = (
                ((x * qfeat) @ query_resid_left.to(device=x.device, dtype=x.dtype))
                @ query_resid_right.to(device=x.device, dtype=x.dtype)
                + ((aux_input * qfeat) @ query_aux_resid_left.to(device=x.device, dtype=x.dtype))
                @ query_aux_resid_right.to(device=x.device, dtype=x.dtype)
            )
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
        if self.config.quantization_correction in {"bridge_low_rank_bank", "bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank"}:
            if aux_input is None:
                raise ValueError(f"{self.config.quantization_correction} quantization correction requires aux_input")
            if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" and runtime_query_features is None:
                raise ValueError("bridge_ridge_qk_cab_bank quantization correction requires runtime_query_features")
            base = x
            if self.config.quantization_correction in {"bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank"}:
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
            if self.config.quantization_correction == "bridge_ridge_qk_cab_bank":
                qfeat = runtime_query_features.to(device=x.device, dtype=x.dtype)
                if x.ndim == 2:
                    if qfeat.ndim == 3 and qfeat.shape[1] == 1:
                        qfeat = qfeat.squeeze(1)
                    if qfeat.ndim != 2 or qfeat.shape != x.shape:
                        raise ValueError(
                            "runtime_query_features must align with calibration bridge samples for bridge_ridge_qk_cab_bank, "
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
                            "runtime_query_features must align with translated samples for bridge_ridge_qk_cab_bank, "
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
                if self.config.quantization_correction == "bridge_ridge_qk_cab_bank" and qfeat is not None:
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
        if self.config.use_whitening:
            K_s_rot = apply_whitening(
                K_s_rot,
                self.whiten_K_src[tgt_layer_idx],
                self.whiten_K_mean[tgt_layer_idx],
            )
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
                    runtime_profile=runtime_attention_profile,
                    runtime_query_features=runtime_query_features,
                )
                V_t_rot = self._apply_quantization_correction(
                    V_q,
                    tgt_layer_idx,
                    "V",
                    aux_input=V_pred,
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

        if self.config.use_target_whitening:
            K_t_rot = undo_whitening(
                K_t_rot,
                self.whiten_K_tgt_inv[tgt_layer_idx],
                self.whiten_K_tgt_mean[tgt_layer_idx],
            )
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
            if self.config.use_whitening:
                if grouped_alignment:
                    W_zca_k, mean_k = self._fit_grouped_whitening(Xk, use_target=False)
                    W_zca_v, mean_v = self._fit_grouped_whitening(Xv, use_target=False)
                else:
                    W_zca_k, mean_k = fit_zca_whitening(Xk)
                    W_zca_v, mean_v = fit_zca_whitening(Xv)
                self.whiten_K_src[tgt_l].data.copy_(W_zca_k)
                self.whiten_V_src[tgt_l].data.copy_(W_zca_v)
                self.whiten_K_mean[tgt_l].data.copy_(mean_k)
                self.whiten_V_mean[tgt_l].data.copy_(mean_v)
                Xk = apply_whitening(Xk, W_zca_k, mean_k)
                Xv = apply_whitening(Xv, W_zca_v, mean_v)

            if self.config.use_target_whitening:
                if grouped_alignment:
                    W_tgt_k, mean_tgt_k = self._fit_grouped_whitening(Yk, use_target=True)
                    W_tgt_v, mean_tgt_v = self._fit_grouped_whitening(Yv, use_target=True)
                else:
                    W_tgt_k, mean_tgt_k = fit_zca_whitening(Yk)
                    W_tgt_v, mean_tgt_v = fit_zca_whitening(Yv)
                self.whiten_K_tgt[tgt_l].data.copy_(W_tgt_k)
                self.whiten_V_tgt[tgt_l].data.copy_(W_tgt_v)
                self.whiten_K_tgt_inv[tgt_l].data.copy_(torch.linalg.pinv(W_tgt_k).to(dtype=W_tgt_k.dtype))
                self.whiten_V_tgt_inv[tgt_l].data.copy_(torch.linalg.pinv(W_tgt_v).to(dtype=W_tgt_v.dtype))
                self.whiten_K_tgt_mean[tgt_l].data.copy_(mean_tgt_k)
                self.whiten_V_tgt_mean[tgt_l].data.copy_(mean_tgt_v)
                Yk_fit = apply_whitening(Yk, W_tgt_k, mean_tgt_k)
                Yv_fit = apply_whitening(Yv, W_tgt_v, mean_tgt_v)
            else:
                Yk_fit = Yk
                Yv_fit = Yv

            if grouped_alignment:
                if self.config.alignment_method in {"grouped_transport", "grouped_permutation", "grouped_signature_transport", "grouped_subspace_transport", "grouped_canonical_transport", "grouped_covariance_transport", "grouped_template_transport", "grouped_qk_retrieval_transport", "grouped_contrastive_template_transport", "grouped_template_subspace_transport"}:
                    W_K, plan_k = self._fit_group_transport_alignment(
                        Xk,
                        Yk_fit,
                        lam=self.config.ridge_lambda,
                        residual_rank=self.config.transport_residual_rank,
                        src_layer_idx=src_l,
                        tgt_layer_idx=tgt_l,
                    )
                    W_V, plan_v = self._fit_group_transport_alignment(
                        Xv,
                        Yv_fit,
                        lam=self.config.ridge_lambda,
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
                        lam=self.config.ridge_lambda,
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
                        lam=self.config.ridge_lambda,
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
                        lam=self.config.ridge_lambda,
                        rank=self.config.alignment_rank,
                    )
                    W_V = self._fit_grouped_alignment(
                        Xv,
                        Yv_fit,
                        method=self.config.alignment_method,
                        lam=self.config.ridge_lambda,
                        rank=self.config.alignment_rank,
                    )
            else:
                W_K = fit_alignment(
                    Xk,
                    Yk_fit,
                    method=self.config.alignment_method,
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
                W_V = fit_alignment(
                    Xv,
                    Yv_fit,
                    method=self.config.alignment_method,
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
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
            elif self.config.quantization_correction in {"bridge_ridge", "bridge_ridge_query", "bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_weighted", "bridge_ridge_qk_projector", "bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter"}:
                sample_weights = None
                if self.config.quantization_correction == "bridge_ridge_qk_weighted":
                    if self._bridge_sample_weights is None:
                        raise ValueError(
                            "bridge_ridge_qk_weighted requires bridge sample weights; "
                            "call set_bridge_sample_weights() before fit_from_pairs"
                        )
                    sample_weights = self._bridge_sample_weights[tgt_l].to(device=Xk.device)
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
                    if self.config.quantization_correction in {"bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter"}:
                        if self._bridge_sample_query_features is None:
                            raise ValueError(
                                f"{self.config.quantization_correction} requires bridge sample query features; "
                                "call set_bridge_sample_query_features() before fit_from_pairs"
                            )
                        query_features = self._bridge_sample_query_features[tgt_l].to(device=Xk.device)
                        base_k = K_quant @ proj_k + K_pred @ aux_proj_k + bias_k
                        resid_target_k = Yk_fit - base_k
                        sample_prompt_ids = None
                        if self.config.quantization_correction in {"bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter"}:
                            if self._bridge_sample_prompt_ids is None:
                                raise ValueError(
                                    f"{self.config.quantization_correction} requires bridge sample prompt ids; "
                                    "call set_bridge_sample_prompt_ids() before fit_from_pairs"
                                )
                            sample_prompt_ids = self._bridge_sample_prompt_ids.to(device=Xk.device)
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
                            sample_prompt_ids=sample_prompt_ids,
                        )
                        self.quant_query_resid_K_left[tgt_l].data.copy_(left_k.to(self.quant_query_resid_K_left[tgt_l].dtype))
                        self.quant_query_resid_K_right[tgt_l].data.copy_(right_k.to(self.quant_query_resid_K_right[tgt_l].dtype))
                        self.quant_query_aux_resid_K_left[tgt_l].data.copy_(aux_left_k.to(self.quant_query_aux_resid_K_left[tgt_l].dtype))
                        self.quant_query_aux_resid_K_right[tgt_l].data.copy_(aux_right_k.to(self.quant_query_aux_resid_K_right[tgt_l].dtype))
                self.quant_proj_K[tgt_l].data.copy_(proj_k.to(self.quant_proj_K[tgt_l].dtype))
                self.quant_proj_V[tgt_l].data.copy_(proj_v.to(self.quant_proj_V[tgt_l].dtype))
                self.quant_aux_proj_K[tgt_l].data.copy_(aux_proj_k.to(self.quant_aux_proj_K[tgt_l].dtype))
                self.quant_aux_proj_V[tgt_l].data.copy_(aux_proj_v.to(self.quant_aux_proj_V[tgt_l].dtype))
                self.quant_bias_K[tgt_l].data.copy_(bias_k.to(self.quant_bias_K[tgt_l].dtype))
                self.quant_bias_V[tgt_l].data.copy_(bias_v.to(self.quant_bias_V[tgt_l].dtype))
                if self.config.quantization_correction in {"bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank"}:
                    if self._bridge_prompt_cluster_labels is None or self._bridge_sample_prompt_ids is None:
                        raise ValueError(
                            "bridge_ridge_residual_bank requires bridge template-bank metadata; "
                            "call set_bridge_runtime_template_bank() before fit_from_pairs"
                        )
                    prompt_ids = self._bridge_sample_prompt_ids.to(device=Xk.device)
                    cluster_labels = self._bridge_prompt_cluster_labels[tgt_l].to(device=Xk.device, dtype=torch.long)
                    sample_expert_ids = cluster_labels[prompt_ids]
                    rank = self.config.quantization_correction_rank
                    base_k = K_quant @ proj_k + K_pred @ aux_proj_k + bias_k
                    base_v = V_quant @ proj_v + V_pred @ aux_proj_v + bias_v
                    residual_target_k = Yk_fit - base_k
                    residual_target_v = Yv_fit - base_v
                    if self.config.quantization_correction == "bridge_ridge_qk_cab_bank":
                        if self._bridge_sample_query_features is None:
                            raise ValueError(
                                "bridge_ridge_qk_cab_bank requires bridge sample query features; "
                                "call set_bridge_sample_query_features() before fit_from_pairs"
                            )
                        query_features = self._bridge_sample_query_features[tgt_l].to(device=Xk.device)
                        global_left_k, global_right_k, global_aux_left_k, global_aux_right_k = self._fit_bridge_query_residual_adapter(
                            K_quant,
                            K_pred,
                            query_features,
                            base_k,
                            residual_target_k,
                            rank=int(self.config.quantization_correction_rank or 8),
                            local_attention_weight=0.25,
                            sample_prompt_ids=prompt_ids,
                        )
                        global_left_v, global_right_v, global_aux_left_v, global_aux_right_v = self._fit_bridge_query_residual_adapter(
                            V_quant,
                            V_pred,
                            query_features,
                            base_v,
                            residual_target_v,
                            rank=int(self.config.quantization_correction_rank or 8),
                            local_attention_weight=0.25,
                            sample_prompt_ids=prompt_ids,
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
                    min_samples = max(8, int(rank or 8))
                    for expert_idx in range(self.config.bridge_bank_size):
                        mask = sample_expert_ids == expert_idx
                        if int(mask.sum().item()) >= min_samples:
                            if self.config.quantization_correction == "bridge_ridge_qk_cab_bank":
                                prompt_ids_mask = prompt_ids[mask]
                                left_k, right_k, aux_left_k, aux_right_k = self._fit_bridge_query_residual_adapter(
                                    K_quant[mask],
                                    K_pred[mask],
                                    query_features[mask],
                                    base_k[mask],
                                    residual_target_k[mask],
                                    rank=int(self.config.quantization_correction_rank or 8),
                                    local_attention_weight=0.25,
                                    sample_prompt_ids=prompt_ids_mask,
                                )
                                left_v, right_v, aux_left_v, aux_right_v = self._fit_bridge_query_residual_adapter(
                                    V_quant[mask],
                                    V_pred[mask],
                                    query_features[mask],
                                    base_v[mask],
                                    residual_target_v[mask],
                                    rank=int(self.config.quantization_correction_rank or 8),
                                    local_attention_weight=0.25,
                                    sample_prompt_ids=prompt_ids_mask,
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
                        if self.config.quantization_correction == "bridge_ridge_qk_cab_bank":
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
                        if self.config.quantization_correction == "bridge_ridge_qk_cab_bank":
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
                        if self.config.quantization_correction == "bridge_ridge_qk_cab_bank":
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
                        if self.config.quantization_correction == "bridge_ridge_qk_cab_bank":
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
            if self.config.quantization_correction in {"bridge_ridge_qk_projector", "bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_cab_bank"}:
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
                runtime_query_features=fit_runtime_query_features,
            )
            V_runtime = V_quant if self.config.quantization_correction == "none" else self._apply_quantization_correction(
                V_quant,
                tgt_l,
                "V",
                aux_input=V_pred,
                runtime_query_features=fit_runtime_query_features,
            )
            if self.config.use_target_whitening:
                K_runtime = undo_whitening(
                    K_runtime,
                    self.whiten_K_tgt_inv[tgt_l],
                    self.whiten_K_tgt_mean[tgt_l],
                )
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
            if self.config.use_target_whitening:
                Yk_hat = Xk @ W_K
                Yv_hat = Xv @ W_V
                q_k["original_space_relative_frobenius_error"] = float(
                    (undo_whitening(Yk_hat, self.whiten_K_tgt_inv[tgt_l], self.whiten_K_tgt_mean[tgt_l]) - Yk).norm()
                    / (Yk.norm() + 1e-12)
                )
                q_v["original_space_relative_frobenius_error"] = float(
                    (undo_whitening(Yv_hat, self.whiten_V_tgt_inv[tgt_l], self.whiten_V_tgt_mean[tgt_l]) - Yv).norm()
                    / (Yv.norm() + 1e-12)
                )
            diagnostics[tgt_l] = {"K": q_k, "V": q_v, "src_layer": src_l}
            if grouped_alignment and self.config.alignment_method in {"grouped_transport", "grouped_permutation", "grouped_signature_transport", "grouped_subspace_transport", "grouped_canonical_transport", "grouped_covariance_transport", "grouped_template_transport", "grouped_qk_retrieval_transport", "grouped_contrastive_template_transport", "grouped_template_subspace_transport"}:
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
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
                WgV = fit_alignment(
                    Xv[:, src_slice],
                    Yv_fit[:, tgt_slice],
                    method=base_method,
                    lam=self.config.ridge_lambda,
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
