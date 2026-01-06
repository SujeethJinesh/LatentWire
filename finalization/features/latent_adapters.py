"""Latent adapter registry feature."""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

from latentwire.models import LMWrapper


class LatentAdaptersFeature:
    """Registers multi-depth latent adapters and exposes optimizer/state hooks."""

    name = "latent_adapters"

    def __init__(self):
        self._param_bank: Dict[str, List[nn.Parameter]] = {}
        self._summaries: Dict[str, Dict[str, Any]] = {}
        self._lr: float = 1e-4

    def _validate_wrapper(self, name: str, wrapper: LMWrapper) -> None:
        if not wrapper.use_latent_adapters:
            return
        if not hasattr(wrapper, "latent_adapters") or len(wrapper.latent_adapters) == 0:
            raise RuntimeError(
                f"Latent adapters requested for {name} but wrapper.latent_adapters is empty. "
                "Ensure LMConfig.latent_adapter_layers is non-empty and matches the checkpoint."
            )
        params = [p for p in wrapper.latent_adapters.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                f"{name}: latent adapters exist but expose no trainable parameters. "
                "Verify adapter initialization and requires_grad flags."
            )

    def apply_post_model_build(self, context, wrappers: Dict[str, LMWrapper], extra_params):
        args = context.args
        self._param_bank.clear()
        self._summaries.clear()
        self._lr = float(getattr(args, "lr", 1e-4))

        requested = bool(getattr(args, "use_latent_adapters", False))
        for name, wrapper in wrappers.items():
            if wrapper is None:
                continue
            if wrapper.use_latent_adapters:
                self._validate_wrapper(name, wrapper)
                params = [p for p in wrapper.latent_adapters.parameters() if p.requires_grad]
                self._param_bank[name] = params
                self._summaries[name] = {
                    "layers": list(wrapper.latent_adapter_layers),
                    "num_params": sum(p.numel() for p in params),
                    "num_tensors": len(params),
                    "dropout": float(getattr(wrapper.cfg, "latent_adapter_dropout", 0.0)),
                    "heads": int(getattr(wrapper.cfg, "latent_adapter_heads", 0)),
                }
            else:
                self._summaries[name] = {
                    "layers": [],
                    "num_params": 0,
                    "num_tensors": 0,
                    "dropout": None,
                    "heads": None,
                }

        if requested and not self._param_bank:
            raise RuntimeError("use_latent_adapters=True but no wrappers contributed latent adapter parameters.")

    def optimizer_param_groups(self) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for _, params in self._param_bank.items():
            if not params:
                continue
            groups.append({"params": params, "lr": self._lr})
        return groups

    def metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"latent_adapters_enabled": bool(self._param_bank)}
        for name, summary in self._summaries.items():
            metrics[f"latent_adapters_params_{name}"] = summary["num_params"]
            if summary["layers"]:
                metrics[f"latent_adapters_layers_{name}"] = ",".join(str(x) for x in summary["layers"])
        return metrics

    def state(self) -> Dict[str, Any]:
        return {
            "latent_adapter_params": self._param_bank,
            "latent_adapter_summaries": self._summaries,
        }
