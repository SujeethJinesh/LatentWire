"""Latent coprocessor feature hook."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from latentwire.models import LatentCoprocessor, LMWrapper


class CoprocessorFeature:
    """Builds latent coprocessors that write KV deltas into decoder caches."""

    name = "coprocessor"

    def __init__(self):
        self.coprocessors: Dict[str, LatentCoprocessor] = {}
        self._lr: float = 1e-4
        self._summaries: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _primary_device(wrapper: LMWrapper) -> torch.device:
        return next(wrapper.model.parameters()).device

    @staticmethod
    def _head_overrides(spec: str, default: int, num_layers: int) -> List[int]:
        if not spec:
            return [default] * num_layers
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            return [default] * num_layers
        values: List[int] = []
        for raw in parts:
            try:
                values.append(int(raw))
            except ValueError:
                values.append(default)
        if len(values) == 1:
            values = values * num_layers
        if len(values) < num_layers:
            values.extend([values[-1]] * (num_layers - len(values)))
        return values[:num_layers]

    def _load_saved_state(self, name: str, module: LatentCoprocessor, context) -> None:
        state_map = context.get("coprocessor_state", None)
        if not state_map:
            return
        state = state_map.get(name)
        if state is None:
            return
        try:
            module.load_state_dict(state, strict=True)
            print(f"[Coprocessor] Restored state for {name} from checkpoint blob.")
        except Exception as exc:
            print(f"[Coprocessor] Warning: failed to restore state for {name}: {exc}")

    def apply_post_model_build(self, context, wrappers: Dict[str, LMWrapper], extra_params: Dict[str, List[nn.Parameter]]) -> None:
        args = context.args

        if getattr(args, "use_deep_prefix", False) and getattr(args, "use_coprocessor", False):
            raise ValueError("Cannot enable both deep prefix and coprocessor simultaneously.")

        if not getattr(args, "use_coprocessor", False):
            return

        kv_len = max(int(getattr(args, "coprocessor_len", 1)), 1)
        width = int(getattr(args, "coprocessor_width", 256))
        dropout = float(getattr(args, "coprocessor_dropout", 0.0))
        kv_scale = float(getattr(args, "coprocessor_kv_scale", 0.8))
        pool = str(getattr(args, "coprocessor_pool", "mean"))
        override_spec = str(getattr(args, "coprocessor_heads", "") or "")

        self.coprocessors.clear()
        self._summaries.clear()
        self._lr = float(getattr(args, "lr", 1e-4))

        for name, wrapper in wrappers.items():
            if wrapper is None or wrapper.model is None:
                continue
            cfg = wrapper.model.config
            num_layers = getattr(cfg, "num_hidden_layers", None)
            num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
            num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
            if num_layers is None or num_heads is None or num_kv_heads is None:
                print(f"[Coprocessor] ⚠️  Skipping {name}: missing layer/head metadata")
                continue
            head_dim = wrapper.d_model // int(num_heads)
            heads_per_layer = self._head_overrides(override_spec, int(num_kv_heads), int(num_layers))

            module = LatentCoprocessor(
                d_z=args.d_z,
                heads_per_layer=heads_per_layer,
                head_dim=head_dim,
                kv_len=kv_len,
                width=width,
                dropout=dropout,
                kv_scale=kv_scale,
                pool=pool,
            ).to(self._primary_device(wrapper))
            module.train()
            self._load_saved_state(name, module, context)
            self.coprocessors[name] = module
            params = [p for p in module.parameters() if p.requires_grad]
            extra_params.setdefault(name, []).extend(params)

            self._summaries[name] = {
                "kv_len": kv_len,
                "width": width,
                "dropout": dropout,
                "kv_scale": kv_scale,
                "heads": heads_per_layer,
                "layers": int(num_layers),
                "pool": pool,
                "params": sum(p.numel() for p in params),
            }

    def optimizer_param_groups(self) -> List[Dict[str, Any]]:
        params = [
            p for module in self.coprocessors.values()
            for p in module.parameters() if p.requires_grad
        ]
        if not params:
            return []
        return [{"params": params, "lr": self._lr}]

    def metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"coprocessor_enabled": bool(self.coprocessors)}
        for name, summary in self._summaries.items():
            metrics[f"coprocessor_params_{name}"] = summary["params"]
            metrics[f"coprocessor_kv_len_{name}"] = summary["kv_len"]
            metrics[f"coprocessor_heads_{name}"] = ",".join(str(h) for h in summary["heads"])
        return metrics

    def state(self) -> Dict[str, Any]:
        return {
            "coprocessors": self.coprocessors,
            "coprocessor_summaries": self._summaries,
        }
