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
    #                 | 'grouped_template_subspace_transport'
    #                 | 'broadcast_template_transport'
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
    # learns a diagonal scale+bias, while `ridge` learns a full linear map
    # plus bias in rotated target space.
    quantization_correction: str = "none"

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
        # Calibration-time grouped attention templates are used only while
        # fitting grouped template transport and are not checkpoint state.
        self._transport_src_group_templates: list[torch.Tensor] | None = None
        self._transport_tgt_group_templates: list[torch.Tensor] | None = None
        self._broadcast_transport_plan_K: list[torch.Tensor] | None = None
        self._broadcast_transport_plan_V: list[torch.Tensor] | None = None

        # --- Optional pre-quant denoising filters and quantization repair ---
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
        self.quant_proj_K = nn.ParameterList(
            [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_proj_V = nn.ParameterList(
            [nn.Parameter(torch.eye(self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_bias_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_bias_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
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
        if self.config.alignment_method in {"grouped_template_transport", "grouped_template_subspace_transport"} and signature_weight > 0.0:
            if self._transport_src_group_templates is None or self._transport_tgt_group_templates is None:
                raise ValueError("grouped_template_transport requires calibration-time group templates")
            src_templates = self._transport_src_group_templates[src_layer_idx].to(device=X.device, dtype=X.dtype)
            tgt_templates = self._transport_tgt_group_templates[tgt_layer_idx].to(device=X.device, dtype=X.dtype)
            if src_templates.shape[0] != group_count or tgt_templates.shape[0] != group_count:
                raise ValueError(
                    "grouped template counts must match the transport group count, "
                    f"got {tuple(src_templates.shape)} and {tuple(tgt_templates.shape)} for {group_count} groups"
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
                elif self.config.alignment_method == "grouped_template_subspace_transport" and signature_weight > 0.0:
                    y_hat = X[:, src_slice] @ W_block
                    template_dist = self._template_distance(src_templates[src_idx], tgt_templates[tgt_idx])
                    subspace_dist = self._subspace_distance(y_hat, Y[:, tgt_slice])
                    score = score - signature_weight * float(template_dist + subspace_dist)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._transport_src_group_templates is None or self._transport_tgt_group_templates is None:
            raise ValueError("broadcast_template_transport requires calibration-time head templates")
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
                    template_dist = self._template_distance(src_templates[src_idx], tgt_templates[tgt_idx])
                    score = score - weight * float(template_dist)
                scores[src_idx, tgt_idx] = score
                blocks[(src_idx, tgt_idx)] = W_block
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

    def _apply_quantization_correction(self, x: torch.Tensor, tgt_layer_idx: int, kind: str) -> torch.Tensor:
        if self.config.quantization_correction == "none":
            return x
        if kind == "K":
            scale = self.quant_scale_K[tgt_layer_idx]
            proj = self.quant_proj_K[tgt_layer_idx]
            bias = self.quant_bias_K[tgt_layer_idx]
        elif kind == "V":
            scale = self.quant_scale_V[tgt_layer_idx]
            proj = self.quant_proj_V[tgt_layer_idx]
            bias = self.quant_bias_V[tgt_layer_idx]
        else:
            raise ValueError(f"Unknown correction kind: {kind}")
        if self.config.quantization_correction == "affine":
            return x * scale.to(device=x.device, dtype=x.dtype) + bias.to(device=x.device, dtype=x.dtype)
        if self.config.quantization_correction == "ridge":
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
            K_q = self.quantizer.quantize_dequantize(K_t_rot)
            V_q = self.quantizer.quantize_dequantize(V_t_rot)
            if quantization_control == "real":
                K_t_rot = self._apply_quantization_correction(K_q, tgt_layer_idx, "K")
                V_t_rot = self._apply_quantization_correction(V_q, tgt_layer_idx, "V")
            elif quantization_control == "matched_noise":
                def add_matched_noise(x: torch.Tensor, q: torch.Tensor, salt: int) -> torch.Tensor:
                    q = self._apply_quantization_correction(q, tgt_layer_idx, "K" if salt == 0 else "V")
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
        a_k = self._effective_gate(
            torch.sigmoid(self.gate_K[tgt_layer_idx]),
            K_t_selected,
            K_t_hat_selected,
            fusion_rule,
        )
        a_v = self._effective_gate(
            torch.sigmoid(self.gate_V[tgt_layer_idx]),
            V_t_selected,
            V_t_hat_selected,
            fusion_rule,
        )
        K_t_hat = self.apply_head_selection(K_t_hat, tgt_layer_idx, fill=K_t)
        V_t_hat = self.apply_head_selection(V_t_hat, tgt_layer_idx, fill=V_t)
        K_out = (1.0 - a_k) * K_t + a_k * K_t_hat
        V_out = (1.0 - a_v) * V_t + a_v * V_t_hat
        return K_out, V_out

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
            self.config.alignment_method == "broadcast_template_transport"
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
                if self.config.alignment_method in {"grouped_transport", "grouped_permutation", "grouped_signature_transport", "grouped_subspace_transport", "grouped_canonical_transport", "grouped_covariance_transport", "grouped_template_transport", "grouped_template_subspace_transport"}:
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
                elif self.config.alignment_method == "broadcast_template_transport":
                    W_K, plan_k = self._fit_broadcast_template_transport_alignment(
                        Xk,
                        Yk_fit,
                        lam=self.config.ridge_lambda,
                        residual_rank=self.config.transport_residual_rank,
                        src_layer_idx=src_l,
                        tgt_layer_idx=tgt_l,
                    )
                    W_V, plan_v = self._fit_broadcast_template_transport_alignment(
                        Xv,
                        Yv_fit,
                        lam=self.config.ridge_lambda,
                        residual_rank=self.config.transport_residual_rank,
                        src_layer_idx=src_l,
                        tgt_layer_idx=tgt_l,
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
            self.quant_proj_K[tgt_l].data.copy_(torch.eye(self.d_t, dtype=self.quant_proj_K[tgt_l].dtype))
            self.quant_proj_V[tgt_l].data.copy_(torch.eye(self.d_t, dtype=self.quant_proj_V[tgt_l].dtype))
            self.quant_bias_K[tgt_l].data.zero_()
            self.quant_bias_V[tgt_l].data.zero_()
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
            elif self.config.quantization_correction == "ridge":
                proj_k, bias_k = self._fit_ridge_correction(K_quant, Yk_fit, lam=self.config.ridge_lambda)
                proj_v, bias_v = self._fit_ridge_correction(V_quant, Yv_fit, lam=self.config.ridge_lambda)
                self.quant_proj_K[tgt_l].data.copy_(proj_k.to(self.quant_proj_K[tgt_l].dtype))
                self.quant_proj_V[tgt_l].data.copy_(proj_v.to(self.quant_proj_V[tgt_l].dtype))
                self.quant_bias_K[tgt_l].data.copy_(bias_k.to(self.quant_bias_K[tgt_l].dtype))
                self.quant_bias_V[tgt_l].data.copy_(bias_v.to(self.quant_bias_V[tgt_l].dtype))
            elif self.config.quantization_correction != "none":
                raise ValueError(f"Unknown quantization_correction: {self.config.quantization_correction}")

            K_runtime = K_quant if self.config.quantization_correction == "none" else self._apply_quantization_correction(
                K_quant,
                tgt_l,
                "K",
            )
            V_runtime = V_quant if self.config.quantization_correction == "none" else self._apply_quantization_correction(
                V_quant,
                tgt_l,
                "V",
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
            if grouped_alignment and self.config.alignment_method in {"grouped_transport", "grouped_permutation", "grouped_signature_transport", "grouped_subspace_transport", "grouped_canonical_transport", "grouped_covariance_transport", "grouped_template_transport", "grouped_template_subspace_transport"}:
                diagnostics[tgt_l]["K_transport_plan"] = self.transport_plan_K[tgt_l].detach().cpu().tolist()
                diagnostics[tgt_l]["V_transport_plan"] = self.transport_plan_V[tgt_l].detach().cpu().tolist()
            elif self.config.alignment_method == "broadcast_template_transport" and self._broadcast_transport_plan_K is not None and self._broadcast_transport_plan_V is not None:
                diagnostics[tgt_l]["K_transport_plan"] = self._broadcast_transport_plan_K[tgt_l].detach().cpu().tolist()
                diagnostics[tgt_l]["V_transport_plan"] = self._broadcast_transport_plan_V[tgt_l].detach().cpu().tolist()

            # Fit optional head-group saliency from local aligned slices.
            group_scores: list[tuple[float, int]] = []
            base_method = self.config.alignment_method.removeprefix("grouped_")
            if base_method in {"transport", "permutation", "signature_transport", "subspace_transport", "canonical_transport", "covariance_transport", "template_transport", "template_subspace_transport", "broadcast_template_transport"}:
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
