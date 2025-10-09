"""Deep prefix feature hook."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from latentwire.models import DeepPrefixGenerator, LMWrapper


class DeepPrefixFeature:
    """Constructs and tracks deep prefix generators when enabled."""

    name = "deep_prefix"

    def __init__(self):
        self.generators: Dict[str, DeepPrefixGenerator] = {}
        self._lr: float = 1e-4
        self._summaries: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _primary_device(wrapper: LMWrapper) -> torch.device:
        return next(wrapper.model.parameters()).device

    def _load_saved_state(self, name: str, generator: DeepPrefixGenerator, context) -> None:
        """Optionally restore generator weights from a checkpoint snapshot."""
        state_map = context.get("deep_prefix_state", None)
        if not state_map:
            return
        state_dict = state_map.get(name)
        if not state_dict:
            return
        try:
            generator.load_state_dict(state_dict, strict=True)
            print(f"[DeepPrefix] Restored generator for {name} from supplied state.")
        except Exception as exc:
            print(f"[DeepPrefix] Warning: failed to restore generator for {name}: {exc}")

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
        self.generators.clear()
        self._summaries.clear()

        for name, wrapper in wrappers.items():
            if wrapper is None or wrapper.model is None:
                continue
            num_layers = getattr(wrapper.model.config, "num_hidden_layers", None)
            num_attention_heads = getattr(
                wrapper.model.config, "num_attention_heads", getattr(wrapper.model.config, "n_head", None)
            )
            num_kv_heads = getattr(wrapper.model.config, "num_key_value_heads", None)
            if num_layers is None or num_attention_heads is None:
                print(f"[DeepPrefix] ⚠️  Skipping {name}: model config missing layer/head counts")
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
            self._load_saved_state(name, generator, context)
            self.generators[name] = generator
            param_count = sum(p.numel() for p in generator.parameters() if p.requires_grad)
            self._summaries[name] = {
                "prefix_len": prefix_len,
                "dropout": dropout,
                "layers": int(num_layers),
                "kv_heads": int(num_kv_heads),
                "head_dim": int(head_dim),
                "params": param_count,
            }

        self._lr = float(getattr(args, "lr", 1e-4))

    def optimizer_param_groups(self) -> List[Dict[str, object]]:
        params: List[nn.Parameter] = []
        for generator in self.generators.values():
            params.extend([p for p in generator.parameters() if p.requires_grad])
        if not params:
            return []
        return [{"params": params, "lr": self._lr}]

    def metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"deep_prefix_enabled": bool(self.generators)}
        for name, summary in self._summaries.items():
            metrics[f"deep_prefix_len_{name}"] = summary["prefix_len"]
            metrics[f"deep_prefix_dropout_{name}"] = summary["dropout"]
            metrics[f"deep_prefix_params_{name}"] = summary["params"]
        return metrics

    def state(self) -> Dict[str, Any]:
        return {
            "deep_prefix_generators": self.generators,
            "deep_prefix_summaries": self._summaries,
        }
