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

    # Rotation variant: 'orthogonal' (Haar-uniform) or 'hadamard' (O(d log d))
    rotation_kind: str = "orthogonal"

    # Whitening: if True, fit a per-layer ZCA whitening as a pre-processing
    # step before rotation + alignment. Corrects anisotropic scaling.
    use_whitening: bool = False

    # Alignment solver: 'auto' | 'identity' | 'procrustes'
    #                 | 'procrustes_rand' | 'ridge' | 'cca' | 'reduced_rank'
    alignment_method: str = "auto"
    ridge_lambda: float = 1e-3
    alignment_rank: int | None = None  # for cca / reduced_rank

    # Layer pairing: 'interp', 'cka', or a list of length num_tgt_layers
    layer_pairing: str | list[int] = "interp"
    layer_selection_topk: int | None = None
    layer_selection_ratio: float = 1.0
    layer_selection_metric: str = "mean_cosine_similarity"

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
        self._fitted = False

    @staticmethod
    def _build_layer_map(cfg: TranslatorConfig) -> list[int]:
        if isinstance(cfg.layer_pairing, list):
            assert len(cfg.layer_pairing) == cfg.num_tgt_layers
            return list(cfg.layer_pairing)
        if cfg.layer_pairing in {"interp", "cka"}:
            return [
                min(int(round(i * cfg.num_src_layers / cfg.num_tgt_layers)), cfg.num_src_layers - 1)
                for i in range(cfg.num_tgt_layers)
            ]
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

    def gate_value(self, tgt_layer_idx: int) -> tuple[float, float]:
        return (
            float(torch.sigmoid(self.gate_K[tgt_layer_idx]).detach()),
            float(torch.sigmoid(self.gate_V[tgt_layer_idx]).detach()),
        )

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

    # ------------------------------------------------------------------
    # Translation + fusion
    # ------------------------------------------------------------------

    def translate_layer(
        self,
        K_s: torch.Tensor,
        V_s: torch.Tensor,
        tgt_layer_idx: int,
        quantize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Translate one source layer's KV into the target's KV space.

        Args:
            K_s, V_s: [batch, src_num_heads, seq, src_head_dim]
            tgt_layer_idx: target layer index l_t
            quantize: whether to round-trip through the Lloyd-Max quantizer
                      (simulates the compressed transmission channel)
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

        # 3) Optional: round-trip through Lloyd-Max quantizer. The output is
        #    the dequantized reconstruction — bit-accurate simulation of a
        #    compressed channel, but with a differentiable straight-through
        #    estimator if we wrap in a detach trick (omitted for clarity).
        if quantize:
            K_t_rot = self.quantizer.quantize_dequantize(K_t_rot)
            V_t_rot = self.quantizer.quantize_dequantize(V_t_rot)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated fusion of target's own KV with translated source KV.

        K_final = (1 - alpha) * K_target_own + alpha * K_translated

        Shapes of the two inputs must match. For same-seq-length case this is
        a pointwise blend; for mixed-seq cases the caller is responsible for
        alignment (e.g., by concatenating along seq).
        """
        a_k = torch.sigmoid(self.gate_K[tgt_layer_idx])
        a_v = torch.sigmoid(self.gate_V[tgt_layer_idx])
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
                W_zca_k, mean_k = fit_zca_whitening(Xk)
                W_zca_v, mean_v = fit_zca_whitening(Xv)
                self.whiten_K_src[tgt_l].data.copy_(W_zca_k)
                self.whiten_V_src[tgt_l].data.copy_(W_zca_v)
                self.whiten_K_mean[tgt_l].data.copy_(mean_k)
                self.whiten_V_mean[tgt_l].data.copy_(mean_v)
                Xk = apply_whitening(Xk, W_zca_k, mean_k)
                Xv = apply_whitening(Xv, W_zca_v, mean_v)

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

            q_k = alignment_quality(Xk, Yk, W_K)
            q_v = alignment_quality(Xv, Yv, W_V)
            diagnostics[tgt_l] = {"K": q_k, "V": q_v, "src_layer": src_l}
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
        model.load_state_dict(payload["state_dict"])
        model._fitted = payload.get("fitted", True)
        return model
