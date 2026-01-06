import argparse
import json
import os
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from latentwire.config import TrainingConfig


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_training_config(path: str) -> Tuple[TrainingConfig, Dict[str, Any]]:
    with open(path, "r") as f:
        cfg_dict = json.load(f)
    cfg = TrainingConfig.from_dict(cfg_dict)
    return cfg, cfg.to_dict()


def apply_overrides(config_dict: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """
    Mutate a nested configuration dictionary using overrides of the form
    "section.key=value" or "key=value".
    """
    cfg = deepcopy(config_dict)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be of the form key=value (got {item!r})")
        raw_key, raw_value = item.split("=", 1)
        raw_key = raw_key.strip()
        path = raw_key.split(".")
        cursor = cfg
        for key in path[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        key = path[-1]
        value = _infer_type(raw_value.strip())
        cursor[key] = value
    return cfg


def _infer_type(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except Exception:
            pass
    return value


def flatten_training_config(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for group, payload in cfg_dict.items():
        if group == "evaluation":
            continue
        if isinstance(payload, dict):
            for key, value in payload.items():
                flat[key] = value
        else:
            flat[group] = payload
    return flat


BOOL_ENABLE_FLAGS = {
    "use_lora": "--use_lora",
    "use_prefix": "--use_prefix",
    "use_deep_prefix": "--use_deep_prefix",
    "use_latent_adapters": "--use_latent_adapters",
    "use_gist_head": "--use_gist_head",
    "use_latent_refiner": "--use_latent_refiner",
    "use_chat_template": "--use_chat_template",
    "use_deep_prefix": "--use_deep_prefix",
    "use_latent_refiner": "--use_latent_refiner",
    "grad_ckpt": "--grad_ckpt",
    "use_coprocessor": "--use_coprocessor",
    "auto_resume": "--auto_resume",
    "baseline_verification": "--baseline_verification",
    "debug": "--debug",
    "adapter_colorize": "--adapter_colorize",
    "encoder_use_chat_template": "--encoder_use_chat_template",
    "freeze_encoder": "--freeze_encoder",
    "sequential_models": "--sequential_models",
    "load_4bit": "--load_4bit",
    "skip_prefix_acc": "--skip_prefix_acc",
    "kd_skip_text": "--kd_skip_text",
    "no_load_optimizer": "--no_load_optimizer",
    "no_load_lr_scheduler": "--no_load_lr_scheduler",
    "reset_epoch": "--reset_epoch",
    "save_training_stats": "--save_training_stats",
}

BOOL_DISABLE_FLAGS = {
    "adapter_metadata": "--no_adapter_metadata",
}


def config_to_argv(flat_cfg: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for key, value in flat_cfg.items():
        if value is None:
            continue
        if key == "train_encoder":
            # train_encoder is implied; --freeze_encoder toggles the opposite.
            continue
        if key in BOOL_ENABLE_FLAGS:
            if bool(value):
                argv.append(BOOL_ENABLE_FLAGS[key])
            continue
        if key in BOOL_DISABLE_FLAGS:
            if not bool(value):
                argv.append(BOOL_DISABLE_FLAGS[key])
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
            continue
        if isinstance(value, list):
            value_str = ",".join(str(item) for item in value)
        else:
            value_str = str(value)
        argv.extend([f"--{key}", value_str])
    return argv


def ensure_metrics_history(save_dir: str) -> str:
    _ensure_dir(save_dir)
    path = os.path.join(save_dir, "metrics_history.jsonl")
    if not os.path.exists(path):
        open(path, "a").close()
    return path


def append_metrics_history(
    save_dir: str,
    record: Dict[str, Any],
) -> None:
    path = ensure_metrics_history(save_dir)
    record = dict(record)
    record.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def summarize_features(cfg_dict: Dict[str, Any]) -> Dict[str, bool]:
    features = cfg_dict.get("features", {})
    summary = {k: bool(v) for k, v in features.items()}
    summary["use_gist_head"] = bool(cfg_dict.get("gist", {}).get("gist_weight", 0.0) > 0)
    return summary


class TeeStream:
    """Write text to multiple underlying streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def namespace_from_argv(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("argv", nargs="*")
    return parser.parse_args(["dummy", *argv])
