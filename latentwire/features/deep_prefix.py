"""Deep prefix feature hook."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from latentwire.models import DeepPrefixGenerator, LMWrapper


class DeepPrefixFeature:
    """Constructs deep prefix generators when enabled."""

    name = "deep_prefix"

    def __init__(self):
        self.generators: Dict[str, DeepPrefixGenerator] = {}
        self._lr: float = 1e-4

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _primary_device(wrapper: LMWrapper) -> torch.device:
        return next(wrapper.model.parameters()).device

    def apply_post_model_build(self, context, wrappers, extra_params):
        args = context.args
        shared_len = context.get("latent_shared_len", args.latent_len or 0)
        private_len = context.get("latent_private_len", 0)
        cfg_prefix_len = (
            args.deep_prefix_len
            if args.deep_prefix_len is not None
            else (shared_len + private_len)
        )

        if cfg_prefix_len is None or cfg_prefix_len <= 0:
            raise ValueError("--use_deep_prefix requires deep_prefix_len > 0 or non-zero latent length")

        prefix_len = int(cfg_prefix_len)
        dropout = float(max(args.deep_prefix_dropout, 0.0))

        for name, wrapper in wrappers.items():
            if wrapper is None or wrapper.model is None:
                continue
            num_layers = getattr(wrapper.model.config, "num_hidden_layers", None)
            num_attention_heads = getattr(wrapper.model.config, "num_attention_heads", getattr(wrapper.model.config, "n_head", None))
            num_kv_heads = getattr(wrapper.model.config, "num_key_value_heads", None)
            if num_layers is None or num_attention_heads is None:
                print(f"[WARN] Skipping deep prefix for {name}: model config missing layer/head counts")
                continue
            if num_kv_heads is None:
                num_kv_heads = num_attention_heads
            head_dim = wrapper.d_model // int(num_attention_heads)
            generator = DeepPrefixGenerator(
                d_z=wrapper.d_model,
                prefix_len=prefix_len,
                num_layers=int(num_layers),
                num_kv_heads=int(num_kv_heads),
                head_dim=int(head_dim),
                dropout=dropout,
            ).to(self._primary_device(wrapper))
            generator.train()
            self.generators[name] = generator

        self._lr = getattr(args, "lr", 1e-4)

    def optimizer_param_groups(self) -> List[Dict[str, object]]:
        params: List[nn.Parameter] = []
        for generator in self.generators.values():
            params.extend([p for p in generator.parameters() if p.requires_grad])
        if not params:
            return []
        return [{"params": params, "lr": self._lr}]

    def metrics(self) -> Dict[str, int]:
        return {
            f"deep_prefix_params_{name}": sum(p.numel() for p in gen.parameters())
            for name, gen in self.generators.items()
        }

    def state(self) -> Dict[str, Dict[str, DeepPrefixGenerator]]:
        return {"deep_prefix_generators": self.generators}
