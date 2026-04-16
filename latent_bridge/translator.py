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
from dataclasses import dataclass, asdict
from typing import Sequence

import torch
import torch.nn as nn

from .rotation import make_rotation, fit_zca_whitening, apply_whitening
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

    # Alignment solver: 'auto' | 'identity' | 'procrustes'
    #                 | 'procrustes_rand' | 'ridge' | 'cca' | 'reduced_rank'
    #                 | 'grouped_' + any of the above
    # Grouped variants fit one block per head-group instead of a single flat
    # all-head projection. When src/tgt head counts match, this degenerates to
    # true per-head alignment.
    alignment_method: str = "auto"
    ridge_lambda: float = 1e-3
    alignment_rank: int | None = None  # for cca / reduced_rank

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

    # Optional decoder-side affine correction after quantize/dequantize.
    quantization_correction: str = "none"

    # Fusion rule for combining target and translated K/V. 'static' keeps the
    # checkpointed scalar gates as-is; cosine-based rules attenuate translated
    # KV when its flattened direction disagrees with the target cache.
    fusion_rule: str = "static"

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
        self.quant_bias_K = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
        self.quant_bias_V = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.d_t), requires_grad=False) for _ in range(config.num_tgt_layers)]
        )
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

    def _effective_gate(
        self,
        base_gate: torch.Tensor,
        target_kv: torch.Tensor,
        translated_kv: torch.Tensor,
        fusion_rule: str,
    ) -> torch.Tensor:
        if fusion_rule == "static":
            return base_gate

        cosine = self._flatten_cosine_similarity(target_kv, translated_kv)
        residual = (translated_kv - target_kv).float()
        residual_var = residual.pow(2).mean()
        target_var = target_kv.float().pow(2).mean().clamp_min(1e-8)
        translated_var = translated_kv.float().pow(2).mean().clamp_min(1e-8)
        if fusion_rule == "cosine":
            scale = cosine.clamp(0.0, 1.0)
        elif fusion_rule == "cosine_shifted":
            scale = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
        elif fusion_rule == "js_shrinkage":
            scale = (1.0 - residual_var / translated_var).clamp(0.0, 1.0)
        elif fusion_rule == "kalman":
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

    def _fit_grouped_whitening(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = torch.zeros(self.d_s, self.d_s, dtype=X.dtype, device=X.device)
        mean = torch.zeros(1, self.d_s, dtype=X.dtype, device=X.device)
        for feature_slice in self._group_feature_slices(use_target=False):
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

    def _apply_quantization_correction(self, x: torch.Tensor, tgt_layer_idx: int, kind: str) -> torch.Tensor:
        if self.config.quantization_correction == "none":
            return x
        if self.config.quantization_correction != "affine":
            raise ValueError(f"Unknown quantization_correction: {self.config.quantization_correction}")
        if kind == "K":
            scale = self.quant_scale_K[tgt_layer_idx]
            bias = self.quant_bias_K[tgt_layer_idx]
        elif kind == "V":
            scale = self.quant_scale_V[tgt_layer_idx]
            bias = self.quant_bias_V[tgt_layer_idx]
        else:
            raise ValueError(f"Unknown correction kind: {kind}")
        return x * scale.to(device=x.device, dtype=x.dtype) + bias.to(device=x.device, dtype=x.dtype)

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

        grouped_alignment = self.config.alignment_method.startswith("grouped_")
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
                    W_zca_k, mean_k = self._fit_grouped_whitening(Xk)
                    W_zca_v, mean_v = self._fit_grouped_whitening(Xv)
                else:
                    W_zca_k, mean_k = fit_zca_whitening(Xk)
                    W_zca_v, mean_v = fit_zca_whitening(Xv)
                self.whiten_K_src[tgt_l].data.copy_(W_zca_k)
                self.whiten_V_src[tgt_l].data.copy_(W_zca_v)
                self.whiten_K_mean[tgt_l].data.copy_(mean_k)
                self.whiten_V_mean[tgt_l].data.copy_(mean_v)
                Xk = apply_whitening(Xk, W_zca_k, mean_k)
                Xv = apply_whitening(Xv, W_zca_v, mean_v)

            if grouped_alignment:
                W_K = self._fit_grouped_alignment(
                    Xk,
                    Yk,
                    method=self.config.alignment_method,
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
                W_V = self._fit_grouped_alignment(
                    Xv,
                    Yv,
                    method=self.config.alignment_method,
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
            else:
                W_K = fit_alignment(
                    Xk,
                    Yk,
                    method=self.config.alignment_method,
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
                W_V = fit_alignment(
                    Xv,
                    Yv,
                    method=self.config.alignment_method,
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
            self.W_K[tgt_l].data.copy_(W_K.to(self.W_K[tgt_l].dtype))
            self.W_V[tgt_l].data.copy_(W_V.to(self.W_V[tgt_l].dtype))

            # Fit optional target-space denoising before quantization.
            pre_quant_filter_k = self._fit_pre_quant_filter(Yk)
            pre_quant_filter_v = self._fit_pre_quant_filter(Yv)
            self.pre_quant_filter_K[tgt_l].data.copy_(pre_quant_filter_k.to(self.pre_quant_filter_K[tgt_l].dtype))
            self.pre_quant_filter_V[tgt_l].data.copy_(pre_quant_filter_v.to(self.pre_quant_filter_V[tgt_l].dtype))

            # Optional decoder-side affine correction to counter quantization bias.
            self.quant_scale_K[tgt_l].data.fill_(1.0)
            self.quant_scale_V[tgt_l].data.fill_(1.0)
            self.quant_bias_K[tgt_l].data.zero_()
            self.quant_bias_V[tgt_l].data.zero_()
            if self.config.quantization_correction == "affine":
                K_pred = (Xk @ W_K) @ pre_quant_filter_k
                V_pred = (Xv @ W_V) @ pre_quant_filter_v
                K_quant = self.quantizer.quantize_dequantize(K_pred)
                V_quant = self.quantizer.quantize_dequantize(V_pred)
                scale_k, bias_k = self._fit_affine_correction(K_quant, Yk)
                scale_v, bias_v = self._fit_affine_correction(V_quant, Yv)
                self.quant_scale_K[tgt_l].data.copy_(scale_k.to(self.quant_scale_K[tgt_l].dtype))
                self.quant_scale_V[tgt_l].data.copy_(scale_v.to(self.quant_scale_V[tgt_l].dtype))
                self.quant_bias_K[tgt_l].data.copy_(bias_k.to(self.quant_bias_K[tgt_l].dtype))
                self.quant_bias_V[tgt_l].data.copy_(bias_v.to(self.quant_bias_V[tgt_l].dtype))
            elif self.config.quantization_correction != "none":
                raise ValueError(f"Unknown quantization_correction: {self.config.quantization_correction}")

            q_k = alignment_quality(Xk, Yk, W_K)
            q_v = alignment_quality(Xv, Yv, W_V)
            diagnostics[tgt_l] = {"K": q_k, "V": q_v, "src_layer": src_l}

            # Fit optional head-group saliency from local aligned slices.
            group_scores: list[tuple[float, int]] = []
            base_method = self.config.alignment_method.removeprefix("grouped_")
            for group_idx, (src_slice, tgt_slice) in enumerate(
                zip(self._group_feature_slices(use_target=False), self._group_feature_slices(use_target=True))
            ):
                WgK = fit_alignment(
                    Xk[:, src_slice],
                    Yk[:, tgt_slice],
                    method=base_method,
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
                WgV = fit_alignment(
                    Xv[:, src_slice],
                    Yv[:, tgt_slice],
                    method=base_method,
                    lam=self.config.ridge_lambda,
                    rank=self.config.alignment_rank,
                )
                qg_k = alignment_quality(Xk[:, src_slice], Yk[:, tgt_slice], WgK)
                qg_v = alignment_quality(Xv[:, src_slice], Yv[:, tgt_slice], WgV)
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
