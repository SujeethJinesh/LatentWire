"""Latent adapter feature placeholder."""

from __future__ import annotations

from typing import Any, Dict, List

from latentwire.models import LMWrapper


class LatentAdaptersFeature:
    """Represents multi-depth latent adapters in the registry."""

    name = "latent_adapters"

    def __init__(self):
        self._active_wrappers: List[str] = []

    def apply_post_model_build(self, context, wrappers: Dict[str, LMWrapper], extra_params):
        for name, wrapper in wrappers.items():
            if wrapper is not None and wrapper.use_latent_adapters:
                self._active_wrappers.append(name)

    def optimizer_param_groups(self) -> List[Dict[str, Any]]:
        return []  # No additional parameter groups; handled via wrappers directly.

    def metrics(self) -> Dict[str, Any]:
        return {"latent_adapters_enabled": bool(self._active_wrappers)}

    def state(self) -> Dict[str, Any]:
        return {}

