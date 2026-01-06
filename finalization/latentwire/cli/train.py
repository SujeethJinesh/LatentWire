import argparse
import os
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from latentwire.cli.utils import (
    TeeStream,
    append_metrics_history,
    apply_overrides,
    config_to_argv,
    flatten_training_config,
    load_training_config,
    summarize_features,
)
from latentwire.config import TrainingConfig
from latentwire.cli import eval as eval_cli


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatentWire training CLI")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values (use dot notation, e.g. data.samples=128)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print derived arguments without launching training",
    )
    parser.add_argument(
        "--print-argv",
        action="store_true",
        help="Only print generated argv for latentwire.train and exit",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="run",
        help="Identifier recorded in metrics history entries",
    )
    return parser


def config_to_argv_list(config_path: str, overrides: List[str]) -> Dict[str, Any]:
    cfg_obj, cfg_dict = load_training_config(config_path)
    effective_dict = apply_overrides(cfg_dict, overrides) if overrides else cfg_dict
    cfg = TrainingConfig.from_dict(effective_dict)
    if cfg.system.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.system.cuda_visible_devices
    cfg_dict_for_cli = cfg.to_dict()
    if "system" in cfg_dict_for_cli:
        cfg_dict_for_cli["system"].pop("cuda_visible_devices", None)
    flat = flatten_training_config(cfg_dict_for_cli)
    cfg_dict_clean = cfg_dict_for_cli
    argv = config_to_argv(flat)
    return {
        "config": cfg,
        "config_dict": cfg_dict_clean,
        "argv": argv,
    }


def _eval_settings_to_dict(eval_cfg: Any) -> Dict[str, Any]:
    if eval_cfg is None:
        return {}
    if isinstance(eval_cfg, dict):
        return dict(eval_cfg)
    if is_dataclass(eval_cfg):
        return asdict(eval_cfg)
    raise TypeError(f"Unsupported evaluation config type: {type(eval_cfg)!r}")


def _prepare_eval_config(cfg: TrainingConfig, eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    settings = {k: v for k, v in (eval_cfg or {}).items() if v is not None}

    settings.setdefault("ckpt", cfg.checkpoint.save_dir)
    settings["ckpt"] = str(settings["ckpt"])

    models = settings.get("models") or cfg.model.models or "llama,qwen"
    settings["models"] = models
    model_keys = [m.strip() for m in models.split(",") if m.strip()]

    if "llama" in model_keys and cfg.model.llama_id:
        settings.setdefault("llama_id", cfg.model.llama_id)
    if "qwen" in model_keys and cfg.model.qwen_id:
        settings.setdefault("qwen_id", cfg.model.qwen_id)

    if cfg.model.llama_device_map:
        settings.setdefault("llama_device_map", cfg.model.llama_device_map)
    if cfg.model.qwen_device_map:
        settings.setdefault("qwen_device_map", cfg.model.qwen_device_map)
    if cfg.model.llama_devices:
        settings.setdefault("llama_devices", cfg.model.llama_devices)
    if cfg.model.qwen_devices:
        settings.setdefault("qwen_devices", cfg.model.qwen_devices)

    settings.setdefault("load_4bit", cfg.model.load_4bit)

    dataset = settings.get("dataset") or cfg.data.dataset
    settings["dataset"] = dataset

    settings.setdefault("samples", cfg.evaluation.samples)
    if settings.get("max_new_tokens") is None:
        settings["max_new_tokens"] = cfg.data.max_answer_tokens

    if settings.get("token_budget_k") in (None, 0):
        settings["token_budget_k"] = cfg.encoder.latent_len
    settings.setdefault("token_budget_mode", cfg.evaluation.token_budget_mode)

    if settings.get("hf_encoder_id") is None and cfg.encoder.hf_encoder_id:
        settings["hf_encoder_id"] = cfg.encoder.hf_encoder_id
    if settings.get("max_enc_tokens") is None and cfg.encoder.max_enc_tokens is not None:
        settings["max_enc_tokens"] = cfg.encoder.max_enc_tokens

    if settings.get("out_dir") is None:
        settings["out_dir"] = cfg.checkpoint.save_dir

    if settings.get("sequential_eval") is None:
        settings["sequential_eval"] = bool(cfg.model.sequential_models)
    if settings.get("fresh_eval") is None:
        settings["fresh_eval"] = cfg.evaluation.fresh_eval

    if settings.get("latent_anchor_mode") is None:
        settings["latent_anchor_mode"] = cfg.evaluation.latent_anchor_mode
    if settings.get("latent_anchor_text") is None:
        settings["latent_anchor_text"] = cfg.evaluation.latent_anchor_text

    if settings.get("append_bos_after_prefix") is None:
        anchor_setting = cfg.anchor.train_append_bos_after_prefix
        if anchor_setting in ("auto", "yes", "no"):
            settings["append_bos_after_prefix"] = anchor_setting
        else:
            settings["append_bos_after_prefix"] = "auto"

    if settings.get("use_chat_template") is None:
        chat_pref = (cfg.evaluation.use_chat_template or "yes").lower()
        use_chat = bool(cfg.anchor.use_chat_template) or chat_pref == "yes"
        settings["use_chat_template"] = "yes" if use_chat else "no"

    settings.setdefault("encoder_text_mode", cfg.evaluation.encoder_text_mode)
    settings.setdefault("first_token_top_p", cfg.evaluation.first_token_top_p)
    settings.setdefault("first_token_temperature", cfg.evaluation.first_token_temperature)
    settings.setdefault("chunk_size", cfg.evaluation.chunk_size)
    if settings.get("embedding_replay") is None:
        settings["embedding_replay"] = bool(getattr(cfg.evaluation, "embedding_replay", False))
    if settings.get("embedding_baseline_modes") is None:
        modes = getattr(cfg.evaluation, "embedding_baseline_modes", [])
        if modes:
            settings["embedding_baseline_modes"] = list(modes)

    settings["fresh_eval"] = bool(settings.get("fresh_eval"))
    settings["sequential_eval"] = bool(settings.get("sequential_eval"))

    out_dir = str(settings["out_dir"])
    settings["out_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    return settings


def run_eval_from_config(
    cfg: TrainingConfig,
    eval_cfg: Any,
    tag: str,
    pipeline_log: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run evaluation using the training configuration and return a metrics history record.
    """
    settings = _prepare_eval_config(cfg, _eval_settings_to_dict(eval_cfg))
    argv = eval_cli.dict_to_argv(settings)

    print("=== LatentWire Eval (auto) ===")
    print("Derived eval argv:", " ".join(argv))

    eval_cli.run_eval(argv)

    record_tag = f"{tag}-eval" if tag else "eval"
    record: Dict[str, Any] = {
        "kind": "eval",
        "tag": record_tag,
        "config": deepcopy(settings),
        "argv": list(argv),
    }
    if pipeline_log is not None:
        record["pipeline_log"] = str(pipeline_log)
    return record


def run_train(argv: List[str]) -> None:
    # Check PyTorch before importing train module
    try:
        import torch
    except ImportError as e:
        print("\n" + "="*60)
        print("ERROR: PyTorch is not properly installed or configured.")
        print("="*60)
        print(f"\nOriginal error: {e}")
        print("\nPlease install PyTorch by running:")
        print("  pip install torch torchvision torchaudio")
        print("\nOr visit https://pytorch.org for platform-specific instructions.")
        print("="*60 + "\n")
        sys.exit(1)

    from latentwire import train as train_module

    argv = ["latentwire-train"] + argv
    old_argv = sys.argv
    try:
        sys.argv = argv
        train_module.main()
    finally:
        sys.argv = old_argv


def main(cli_args: Optional[List[str]] = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(cli_args)

    result = config_to_argv_list(args.config, args.override)
    cfg: TrainingConfig = result["config"]
    cfg_dict: Dict[str, Any] = result["config_dict"]
    argv: List[str] = result["argv"]

    run_dir = Path(cfg.checkpoint.save_dir).expanduser().resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pipeline_{args.tag}_{timestamp}.log" if args.tag else f"pipeline_{timestamp}.log"
    log_path = run_dir / log_filename

    feature_summary = summarize_features(cfg_dict)
    pipeline_log: Optional[Path] = None
    eval_record: Optional[Dict[str, Any]] = None

    with ExitStack() as stack:
        if not args.print_argv and not args.dry_run:
            log_file = stack.enter_context(log_path.open("w", encoding="utf-8", buffering=1))
            tee_out = TeeStream(sys.stdout, log_file)
            tee_err = TeeStream(sys.stderr, log_file)
            stack.enter_context(redirect_stdout(tee_out))
            stack.enter_context(redirect_stderr(tee_err))
            pipeline_log = log_path
            print(f"[CLI] Writing pipeline log to {log_path}")

        print("=== LatentWire Train CLI ===")
        print(f"Config: {os.path.abspath(args.config)}")
        if args.override:
            print("Overrides:")
            for item in args.override:
                print(f"  - {item}")
        print("Feature toggles:")
        for name, enabled in feature_summary.items():
            status = "ON" if enabled else "off"
            print(f"  {name}: {status}")
        print("Derived argv:", " ".join(argv))

        if args.print_argv or args.dry_run:
            print("[CLI] Dry run complete; training not launched.")
            pipeline_log = None
            return

        run_train(argv)

        print()
        eval_record = run_eval_from_config(
            cfg,
            cfg.evaluation,
            args.tag or "run",
            pipeline_log=pipeline_log,
        )
        if eval_record is not None:
            eval_record.setdefault("features", feature_summary)

    save_dir = cfg.checkpoint.save_dir
    record = {
        "kind": "train",
        "tag": args.tag,
        "config_path": os.path.abspath(args.config),
        "overrides": deepcopy(args.override),
        "argv": argv,
        "features": feature_summary,
    }
    if pipeline_log is not None:
        record["pipeline_log"] = str(pipeline_log)
    try:
        append_metrics_history(save_dir, record)
        print(f"[CLI] Recorded metrics history entry in {save_dir}")
        if pipeline_log is not None:
            print(f"[CLI] Pipeline log: {pipeline_log}")
    except Exception as exc:
        print(f"[CLI] Warning: failed to append metrics history ({exc})")

    if eval_record is not None:
        eval_out_dir = eval_record.get("config", {}).get("out_dir") or save_dir
        try:
            append_metrics_history(eval_out_dir, eval_record)
            print(f"[CLI] Recorded eval metrics history entry in {eval_out_dir}")
        except Exception as exc:
            print(f"[CLI] Warning: failed to append eval metrics history ({exc})")


if __name__ == "__main__":
    main()
