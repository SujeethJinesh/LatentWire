# latentwire/feature_registry.py
"""
Feature registry and hook system for training.

Each feature can install hooks that manipulate model wrappers, contribute
optimizer parameter groups, or emit diagnostics.  The registry keeps the
training loop decoupled from feature-specific wiring so we can toggle
capabilities independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch.nn as nn

from latentwire.models import LMWrapper, apply_lora_if_requested


@dataclass
class FeatureContext:
    """Context passed to feature hooks."""

    args: Any
    extras: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)


class TrainingFeature:
    """Base class for all training features."""

    name: str = "feature"

    def apply_post_model_build(
        self,
        context: FeatureContext,
        wrappers: Dict[str, LMWrapper],
        extra_params: Dict[str, List[nn.Parameter]],
    ) -> None:
        """Called after LMWrapper objects are constructed."""

    def optimizer_param_groups(self) -> List[Dict[str, Any]]:
        """Return additional optimizer parameter groups."""
        return []

    def metrics(self) -> Dict[str, Any]:
        """Return diagnostic metrics for logging."""
        return {}

    def state(self) -> Dict[str, Any]:
        """Return state to expose to the training loop."""
        return {}


class FeatureRegistry:
    """Registry managing feature hooks."""

    def __init__(self, args: Any):
        self.args = args
        self.extras: Dict[str, Any] = {}
        self.context = FeatureContext(args=args, extras=self.extras)
        self.features: List[TrainingFeature] = []
        self.state: Dict[str, Any] = {}

        # Register built-in features based on CLI flags.
        if getattr(args, "use_lora", False):
            self.features.append(LoRAFeature())
        if getattr(args, "use_deep_prefix", False):
            from latentwire.features import DeepPrefixFeature

            self.features.append(DeepPrefixFeature())
        if getattr(args, "use_latent_adapters", False):
            from latentwire.features import LatentAdaptersFeature

            self.features.append(LatentAdaptersFeature())

        # Placeholder for future features (deep prefix, adapters, coprocessor, ...)

    # ------------------------------------------------------------------ helpers
    def names(self) -> List[str]:
        return [feat.name for feat in self.features]

    def has(self, name: str) -> bool:
        return any(feat.name == name for feat in self.features)

    # ------------------------------------------------------------------ hooks
    def apply_post_model_build(
        self,
        wrappers: Dict[str, LMWrapper],
    ) -> Dict[str, List[nn.Parameter]]:
        extra_params: Dict[str, List[nn.Parameter]] = {
            name: [] for name in wrappers.keys()
        }
        for feature in self.features:
            feature.apply_post_model_build(self.context, wrappers, extra_params)
            self.state.update(feature.state())
        return extra_params

    def optimizer_param_groups(self) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for feature in self.features:
            groups.extend(feature.optimizer_param_groups())
        return groups

    def metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for feature in self.features:
            metrics.update(feature.metrics())
        return metrics

    def set_extra(self, key: str, value: Any) -> None:
        self.extras[key] = value


# ------------------------------------------------------------------------------
# Built-in features


class LoRAFeature(TrainingFeature):
    """Apply LoRA adapters to each wrapper when enabled."""

    name = "lora"

    def __init__(self):
        self._last_param_deltas: Dict[str, int] = {}

    @staticmethod
    def _collect_trainable(module: nn.Module) -> List[nn.Parameter]:
        return [p for p in module.parameters() if p.requires_grad]

    @staticmethod
    def _count_params(module: nn.Module) -> tuple[int, int]:
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        return trainable, total

    def apply_post_model_build(
        self,
        context: FeatureContext,
        wrappers: Dict[str, LMWrapper],
        extra_params: Dict[str, List[nn.Parameter]],
    ) -> None:
        args = context.args
        lora_cfg = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
            "first_n": args.lora_firstN,
        }

        if not wrappers:
            return

        print(f"\n🔧 Applying LoRA (r={lora_cfg['r']}, alpha={lora_cfg['alpha']})...")

        for name, wrapper in wrappers.items():
            model_id = args.llama_id if name == "llama" else args.qwen_id
            if wrapper is None or wrapper.model is None:
                continue

            before_trainable, before_total = self._count_params(wrapper.model)
            print(
                f"   {name.capitalize()} BEFORE LoRA: "
                f"{before_trainable:,} trainable / {before_total:,} total"
            )

            wrapper.model = apply_lora_if_requested(wrapper.model, lora_cfg, model_id)

            after_trainable, after_total = self._count_params(wrapper.model)
            delta = after_trainable - before_trainable
            self._last_param_deltas[name] = delta

            print(
                f"   {name.capitalize()} AFTER LoRA:  "
                f"{after_trainable:,} trainable / {after_total:,} total"
            )
            print(f"   ✓ Added {delta:,} LoRA parameters to {name.capitalize()}")

            extra_params.setdefault(name, []).extend(
                self._collect_trainable(wrapper.model)
            )
            wrapper.model.train()
            wrapper.input_embed = wrapper.model.get_input_embeddings()

    def metrics(self) -> Dict[str, Any]:
        return {f"lora_params_{name}": delta for name, delta in self._last_param_deltas.items()}
