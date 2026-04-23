#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gsm8k_contract_checkpoint_sweep as checkpoint_sweep
from scripts import run_gsm8k_smoke_contract as smoke
from scripts import harness_common as harness


DEFAULT_BASELINE_RESULTS_DIR = "results/gsm8k_smoke_contract_20260421"
DEFAULT_SWEEP_RESULTS_DIR = "results/gsm8k_contract_residual_sweep_20260421"
DEFAULT_CHECKPOINTS_DIR = "checkpoints/gsm8k_contract_residual_sweep_20260421"
DEFAULT_MATERIALIZED_EVAL_FILE = None
DEFAULT_CALIBRATION_FILE = ".debug/calibration_64.txt"

DEFAULT_BASES: dict[str, dict[str, str | int]] = {
    "dynalign_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_module_replace",
        "checkpoint_path": "checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt",
        "existing_rank": 8,
    },
    "dynalign_preserve_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_preserve_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_eigenspace_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_eigenspace_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_saliency_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_saliency_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_saliency_preserve_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_saliency_preserve_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_anchor_tail_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_anchor_tail_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_v8_outlier_escrow_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_routed_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_routed_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_value_routed_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_value_routed_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_query_resampler_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_query_resampler_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_value_bank_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_value_bank_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_value_query_bank_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_value_query_bank_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_value_routed_bank_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_value_routed_bank_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "dynalign_value_verifier_sidecar_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace",
        "checkpoint_path": "",
        "existing_rank": -1,
    },
    "tokenbasis_replace": {
        "quantization_correction": "bridge_ridge_qk_tokenbasis_replace",
        "checkpoint_path": "checkpoints/bridge_ridge_qk_tokenbasis_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_tokenbasis_replace_cal64_chat.pt",
        "existing_rank": 8,
    },
}


@dataclass(frozen=True)
class ResidualSweepConfig:
    source_model: str = smoke.DEFAULT_SOURCE_MODEL
    target_model: str = smoke.DEFAULT_TARGET_MODEL
    eval_file: str = smoke.DEFAULT_EVAL_FILE
    slice_size: int = 32
    materialized_eval_file: str | None = DEFAULT_MATERIALIZED_EVAL_FILE
    baseline_results_dir: str = DEFAULT_BASELINE_RESULTS_DIR
    results_dir: str = DEFAULT_SWEEP_RESULTS_DIR
    checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR
    calibration_file: str = DEFAULT_CALIBRATION_FILE
    device: str = "mps"
    dtype: str = "float32"
    max_new_tokens: int = 64
    gate: float = 0.10
    kv_transport: str = "k_only"
    position_selection_ratio: float = 0.5
    position_selection_metric: str = "attention"
    source_reasoning_mode: str = "brief_analysis"
    use_chat_template: bool = True
    enable_thinking: bool = False
    alignment: str = "grouped_subspace_transport"
    bits: int = 4
    bridge_bank_size: int = 4
    ridge_lambda: float = 1e-3
    fit_ridge_override_lambda: float | None = None
    fit_ridge_override_streams: str = "kv"
    fit_ridge_override_layers: tuple[int, ...] | None = None
    fit_ridge_protected_rank: int | None = None
    transport_residual_rank: int = 4
    transport_temperature: float = 0.1
    transport_sinkhorn_iters: int = 8
    transport_signature_rank: int = 8
    transport_signature_weight: float = 0.0
    whitening: bool = False
    target_whitening: bool = False
    whitening_streams: str = "kv"
    target_whitening_streams: str = "kv"
    conditioning_target_layers: tuple[int, ...] | None = None
    seed: int = 0
    ranks: tuple[int, ...] = (2, 4, 8, 16)
    bases: tuple[str, ...] = ("dynalign_module_replace", "tokenbasis_replace")

    def __post_init__(self) -> None:
        bridge_bank_size = int(self.bridge_bank_size)
        if bridge_bank_size < 0:
            raise ValueError(f"bridge_bank_size must be non-negative, got {bridge_bank_size}")
        object.__setattr__(self, "bridge_bank_size", bridge_bank_size)


def _run(cmd: list[str]) -> None:
    smoke._run(cmd, cwd=ROOT)


def _candidate_label(base_label: str, rank: int, bridge_bank_size: int = 4) -> str:
    bank_suffix = "" if int(bridge_bank_size) == 4 else f"_bank{int(bridge_bank_size)}"
    return f"{base_label}_residrank{rank}{bank_suffix}"


def _float_suffix(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace("+", "").replace(".", "p")


def _conditioning_suffix(config: ResidualSweepConfig) -> str:
    parts: list[str] = []
    if config.whitening:
        parts.append(
            "srcwhite" if config.whitening_streams == "kv" else f"srcwhite{config.whitening_streams}"
        )
    if config.target_whitening:
        parts.append(
            "tgtwhite"
            if config.target_whitening_streams == "kv"
            else f"tgtwhite{config.target_whitening_streams}"
        )
    if config.conditioning_target_layers:
        layer_suffix = "-".join(str(layer) for layer in config.conditioning_target_layers)
        parts.append(f"layers{layer_suffix}")
    if config.fit_ridge_override_lambda is not None:
        streams = config.fit_ridge_override_streams
        if config.fit_ridge_override_layers:
            override_layers = "-".join(str(layer) for layer in config.fit_ridge_override_layers)
            parts.append(f"fitridge{streams}_layers{override_layers}_lam{_float_suffix(config.fit_ridge_override_lambda)}")
        else:
            parts.append(f"fitridge{streams}_lam{_float_suffix(config.fit_ridge_override_lambda)}")
    if config.fit_ridge_protected_rank is not None:
        parts.append(f"protect{int(config.fit_ridge_protected_rank)}")
    if int(config.bridge_bank_size) != 4:
        parts.append(f"bank{int(config.bridge_bank_size)}")
    return "" if not parts else "_" + "_".join(parts)


def _can_reuse_existing_checkpoint(base_label: str, rank: int, config: ResidualSweepConfig) -> bool:
    base_meta = DEFAULT_BASES[base_label]
    return (
        rank == int(base_meta["existing_rank"])
        and int(config.seed) == 0
        and not config.whitening
        and not config.target_whitening
        and config.fit_ridge_override_lambda is None
        and int(config.bridge_bank_size) == 4
    )


def _conditioning_payload(config: ResidualSweepConfig) -> dict[str, Any]:
    return {
        "whitening": bool(config.whitening),
        "target_whitening": bool(config.target_whitening),
        "whitening_streams": config.whitening_streams,
        "target_whitening_streams": config.target_whitening_streams,
        "conditioning_target_layers": list(config.conditioning_target_layers or ()),
        "bridge_bank_size": int(config.bridge_bank_size),
        "fit_ridge_override_lambda": config.fit_ridge_override_lambda,
        "fit_ridge_override_streams": config.fit_ridge_override_streams,
        "fit_ridge_override_layers": list(config.fit_ridge_override_layers or ()),
        "fit_ridge_protected_rank": config.fit_ridge_protected_rank,
    }


def _checkpoint_path(base_label: str, rank: int, config: ResidualSweepConfig) -> pathlib.Path:
    if _can_reuse_existing_checkpoint(base_label, rank, config):
        base_meta = DEFAULT_BASES[base_label]
        return ROOT / str(base_meta["checkpoint_path"])
    seed_suffix = "" if int(config.seed) == 0 else f"_seed{int(config.seed)}"
    conditioning_suffix = _conditioning_suffix(config)
    filename = (
        f"qwen25_to_qwen3_grouped_subspace_transport_w010_r{rank}_"
        f"{base_label}_cal64_chat{conditioning_suffix}{seed_suffix}.pt"
    )
    return ROOT / config.checkpoints_dir / base_label / filename


def _checkpoint_finite_summary(checkpoint_path: pathlib.Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("state_dict", payload)
    tensor_keys = 0
    floating_tensor_keys = 0
    total_numel = 0
    nonfinite_numel = 0
    max_abs = 0.0
    first_bad_key: str | None = None
    nonfinite_keys: list[str] = []
    top_abs_tensors: list[dict[str, Any]] = []
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        tensor_keys += 1
        total_numel += int(value.numel())
        if not value.is_floating_point():
            continue
        floating_tensor_keys += 1
        bad = ~torch.isfinite(value)
        bad_count = int(bad.sum().item())
        nonfinite_numel += bad_count
        if bad_count > 0 and first_bad_key is None:
            first_bad_key = str(key)
            nonfinite_keys.append(str(key))
        elif bad_count > 0:
            nonfinite_keys.append(str(key))
        abs_value = value.abs()
        finite_abs = abs_value[torch.isfinite(abs_value)]
        tensor_max_abs = float(finite_abs.max().item()) if finite_abs.numel() else 0.0
        tensor_mean_abs = float(finite_abs.mean().item()) if finite_abs.numel() else 0.0
        if finite_abs.numel():
            max_abs = max(max_abs, tensor_max_abs)
        top_abs_tensors.append(
            {
                "key": str(key),
                "shape": list(value.shape),
                "max_abs": tensor_max_abs,
                "mean_abs": tensor_mean_abs,
                "nonfinite_numel": bad_count,
            }
        )
    top_abs_tensors.sort(
        key=lambda item: (int(item["nonfinite_numel"] > 0), float(item["max_abs"])),
        reverse=True,
    )
    return {
        "tensor_keys": tensor_keys,
        "floating_tensor_keys": floating_tensor_keys,
        "total_numel": total_numel,
        "nonfinite_numel": nonfinite_numel,
        "max_abs": max_abs,
        "first_bad_key": first_bad_key,
        "nonfinite_keys": nonfinite_keys[:8],
        "top_abs_tensors": top_abs_tensors[:8],
    }


def _checkpoint_health_path(checkpoint_path: pathlib.Path) -> pathlib.Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".health.json")


def _quarantined_checkpoint_path(checkpoint_path: pathlib.Path) -> pathlib.Path:
    return checkpoint_path.with_name(checkpoint_path.stem + ".nonfinite" + checkpoint_path.suffix)


def _write_checkpoint_health(
    *,
    checkpoint_path: pathlib.Path,
    base_label: str,
    rank: int,
    config: ResidualSweepConfig,
    checkpoint_summary: dict[str, Any],
    freshly_created: bool,
    quarantined_path: pathlib.Path | None,
) -> pathlib.Path:
    health_path = _checkpoint_health_path(checkpoint_path)
    payload = {
        "base_label": base_label,
        "residual_rank": rank,
        "seed": int(config.seed),
        "checkpoint_path": str(checkpoint_path),
        "freshly_created": bool(freshly_created),
        "quarantined_checkpoint_path": str(quarantined_path) if quarantined_path is not None else None,
        "conditioning": _conditioning_payload(config),
        **checkpoint_summary,
    }
    smoke._write_json(health_path, payload)
    return health_path


def _validate_checkpoint_finite(checkpoint_path: pathlib.Path) -> dict[str, Any]:
    summary = _checkpoint_finite_summary(checkpoint_path)
    if int(summary["nonfinite_numel"]) > 0:
        raise ValueError(
            "Checkpoint contains non-finite values: "
            f"path={checkpoint_path} nonfinite_numel={summary['nonfinite_numel']} "
            f"first_bad_key={summary['first_bad_key']} max_abs={summary['max_abs']:.4f}"
        )
    return summary


def _safe_checkpoint_summary(checkpoint_path: pathlib.Path) -> dict[str, Any]:
    health_path = _checkpoint_health_path(checkpoint_path)
    if not checkpoint_path.exists():
        if health_path.exists():
            try:
                payload = json.loads(health_path.read_text())
                payload["checkpoint_exists"] = False
                return payload
            except Exception:
                pass
        return {
            "checkpoint_exists": False,
            "tensor_keys": 0,
            "floating_tensor_keys": 0,
            "total_numel": 0,
            "nonfinite_numel": 0,
            "max_abs": 0.0,
            "first_bad_key": None,
            "nonfinite_keys": [],
            "top_abs_tensors": [],
        }
    try:
        return {"checkpoint_exists": True, **_checkpoint_finite_summary(checkpoint_path)}
    except Exception as exc:
        return {
            "checkpoint_exists": True,
            "summary_error": str(exc),
            "tensor_keys": 0,
            "floating_tensor_keys": 0,
            "total_numel": 0,
            "nonfinite_numel": 0,
            "max_abs": 0.0,
            "first_bad_key": None,
            "nonfinite_keys": [],
            "top_abs_tensors": [],
        }


def _failure_status(exc: Exception, checkpoint_summary: dict[str, Any]) -> str:
    if int(checkpoint_summary.get("nonfinite_numel", 0)) > 0:
        return "checkpoint_nonfinite"
    if isinstance(exc, FileNotFoundError):
        return "checkpoint_missing"
    return "candidate_error"


def _failure_row(
    *,
    base_label: str,
    rank: int,
    checkpoint_path: pathlib.Path,
    checkpoint_summary: dict[str, Any],
    config: ResidualSweepConfig,
    error: Exception,
) -> tuple[dict[str, Any], dict[str, bool]]:
    row = {
        "label": _candidate_label(base_label, rank, config.bridge_bank_size),
        "base_label": base_label,
        "residual_rank": rank,
        "bridge_bank_size": int(config.bridge_bank_size),
        "accuracy": 0.0,
        "n": 0,
        "correct": 0,
        "example_ids": [],
        "numeric_extraction_coverage": 0,
        "empty_predictions": config.slice_size,
        "paired_vs_target": {"win": 0, "loss": 0, "tie": 0},
        "seed": int(config.seed),
        "reused_existing_checkpoint": _can_reuse_existing_checkpoint(base_label, rank, config),
        "status": _failure_status(error, checkpoint_summary),
        "failure_reason": str(error),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_nonfinite_numel": int(checkpoint_summary.get("nonfinite_numel", 0)),
        "checkpoint_first_bad_key": checkpoint_summary.get("first_bad_key"),
        "checkpoint_max_abs": float(checkpoint_summary.get("max_abs", 0.0)),
        "checkpoint_health_path": checkpoint_summary.get("health_path"),
        "conditioning": _conditioning_payload(config),
        "checkpoint_summary": checkpoint_summary,
    }
    checks = {
        "row_count_matches_slice": False,
        "example_ids_match_target": False,
        "no_empty_predictions": False,
        "numeric_extraction_coverage": False,
        "beats_target": False,
    }
    return row, checks


def _calibrate_checkpoint(
    *,
    base_label: str,
    rank: int,
    checkpoint_path: pathlib.Path,
    config: ResidualSweepConfig,
) -> dict[str, Any]:
    created_checkpoint = False
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        created_checkpoint = True
        base_meta = DEFAULT_BASES[base_label]
        cmd = [
            harness.python_executable(ROOT),
            str(ROOT / "scripts" / "calibrate.py"),
            "--calibration-file",
            str(ROOT / config.calibration_file),
            "--source-model",
            config.source_model,
            "--target-model",
            config.target_model,
            "--output",
            str(checkpoint_path),
            "--bits",
            str(config.bits),
            "--alignment",
            config.alignment,
            "--ridge-lambda",
            str(config.ridge_lambda),
            "--transport-residual-rank",
            str(config.transport_residual_rank),
            "--transport-temperature",
            str(config.transport_temperature),
            "--transport-sinkhorn-iters",
            str(config.transport_sinkhorn_iters),
            "--transport-signature-rank",
            str(config.transport_signature_rank),
            "--transport-signature-weight",
            str(config.transport_signature_weight),
            "--quantization-correction",
            str(base_meta["quantization_correction"]),
            "--quantization-correction-rank",
            str(rank),
            "--bridge-bank-size",
            str(config.bridge_bank_size),
            "--source-reasoning-mode",
            config.source_reasoning_mode,
            "--device",
            config.device,
            "--dtype",
            config.dtype,
            "--seed",
            str(config.seed),
        ]
        cmd.extend(
            harness.chat_template_cli_args(
                enabled=config.use_chat_template,
                thinking=config.enable_thinking,
            )
        )
        if config.whitening:
            cmd.append("--whitening")
            cmd.extend(["--whitening-streams", config.whitening_streams])
        if config.target_whitening:
            cmd.append("--target-whitening")
            cmd.extend(["--target-whitening-streams", config.target_whitening_streams])
        if config.conditioning_target_layers:
            for layer in config.conditioning_target_layers:
                cmd.extend(["--conditioning-target-layer", str(layer)])
        if config.fit_ridge_override_lambda is not None:
            cmd.extend(["--fit-ridge-override-lambda", str(config.fit_ridge_override_lambda)])
            cmd.extend(["--fit-ridge-override-streams", config.fit_ridge_override_streams])
            for layer in config.fit_ridge_override_layers or ():
                cmd.extend(["--fit-ridge-override-layer", str(layer)])
            if config.fit_ridge_protected_rank is not None:
                cmd.extend(["--fit-ridge-protected-rank", str(config.fit_ridge_protected_rank)])
        _run(cmd)
    checkpoint_summary = _checkpoint_finite_summary(checkpoint_path)
    if int(checkpoint_summary["nonfinite_numel"]) > 0:
        quarantined_path: pathlib.Path | None = None
        if created_checkpoint and checkpoint_path.exists():
            quarantined_path = _quarantined_checkpoint_path(checkpoint_path)
            quarantined_path.parent.mkdir(parents=True, exist_ok=True)
            if quarantined_path.exists():
                quarantined_path.unlink()
            checkpoint_path.replace(quarantined_path)
        health_path = _write_checkpoint_health(
            checkpoint_path=checkpoint_path,
            base_label=base_label,
            rank=rank,
            config=config,
            checkpoint_summary=checkpoint_summary,
            freshly_created=created_checkpoint,
            quarantined_path=quarantined_path,
        )
        raise ValueError(
            "Checkpoint contains non-finite values: "
            f"path={checkpoint_path} nonfinite_numel={checkpoint_summary['nonfinite_numel']} "
            f"first_bad_key={checkpoint_summary['first_bad_key']} max_abs={checkpoint_summary['max_abs']:.4f} "
            f"health_path={health_path}"
        )
    health_path = _write_checkpoint_health(
        checkpoint_path=checkpoint_path,
        base_label=base_label,
        rank=rank,
        config=config,
        checkpoint_summary=checkpoint_summary,
        freshly_created=created_checkpoint,
        quarantined_path=None,
    )
    checkpoint_summary["health_path"] = str(health_path)
    return checkpoint_summary


def _row_checks(row: dict[str, Any], baseline_target_records: list[dict[str, Any]], slice_size: int) -> dict[str, bool]:
    baseline_accuracy = sum(int(r["correct"]) for r in baseline_target_records) / max(len(baseline_target_records), 1)
    return {
        "row_count_matches_slice": row["n"] == slice_size,
        "example_ids_match_target": row["example_ids"] == [r["example_id"] for r in baseline_target_records],
        "no_empty_predictions": row["empty_predictions"] == 0,
        "numeric_extraction_coverage": row["numeric_extraction_coverage"] >= slice_size - 1,
        "beats_target": row["accuracy"] > baseline_accuracy,
    }


def _run_candidate(
    *,
    base_label: str,
    rank: int,
    checkpoint_path: pathlib.Path,
    checkpoint_summary: dict[str, Any],
    config: ResidualSweepConfig,
    materialized_eval_file: pathlib.Path,
    baseline_target_records: list[dict[str, Any]],
    results_dir: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, bool]]:
    label = _candidate_label(base_label, rank, config.bridge_bank_size)
    prediction_output = results_dir / f"{label}.jsonl"
    if not prediction_output.exists():
        cmd = [
            harness.python_executable(ROOT),
            str(ROOT / "latent_bridge" / "evaluate.py"),
            "--translator",
            str(checkpoint_path),
            "--source-model",
            config.source_model,
            "--target-model",
            config.target_model,
            "--eval-file",
            str(materialized_eval_file),
            "--task-type",
            "generation",
            "--device",
            config.device,
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--source-reasoning-mode",
            config.source_reasoning_mode,
            "--kv-transport",
            config.kv_transport,
            "--position-selection-ratio",
            str(config.position_selection_ratio),
            "--position-selection-metric",
            config.position_selection_metric,
            "--gate-mode",
            "fixed",
            "--fixed-gate",
            f"{config.gate:.2f}",
            "--methods",
            "rotalign",
            "--prediction-output",
            str(prediction_output),
            "--random-salt",
            str(config.seed),
        ]
        cmd.extend(
            harness.chat_template_cli_args(
                enabled=config.use_chat_template,
                thinking=config.enable_thinking,
            )
        )
        _run(cmd)

    records = smoke._attach_prompts(smoke._read_jsonl(prediction_output), materialized_eval_file)
    method_records = smoke._group_by_method(records)["rotalign_kv"]
    row = checkpoint_sweep._candidate_row(
        label=label,
        checkpoint_path=checkpoint_path,
        records=method_records,
        baseline_target_records=baseline_target_records,
    )
    row["base_label"] = base_label
    row["residual_rank"] = rank
    row["bridge_bank_size"] = int(config.bridge_bank_size)
    row["seed"] = int(config.seed)
    row["reused_existing_checkpoint"] = _can_reuse_existing_checkpoint(base_label, rank, config)
    row["status"] = "ok"
    row["checkpoint_nonfinite_numel"] = int(checkpoint_summary.get("nonfinite_numel", 0))
    row["checkpoint_first_bad_key"] = checkpoint_summary.get("first_bad_key")
    row["checkpoint_max_abs"] = float(checkpoint_summary.get("max_abs", 0.0))
    row["checkpoint_health_path"] = checkpoint_summary.get("health_path")
    row["prediction_output"] = str(prediction_output)
    row["prediction_meta_output"] = str(prediction_output.with_suffix(prediction_output.suffix + ".meta.json"))
    row["conditioning"] = _conditioning_payload(config)
    row["checkpoint_summary"] = checkpoint_summary
    checks = _row_checks(row, baseline_target_records, config.slice_size)
    return row, checks


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GSM8K32 Residual Rank Sweep",
        "",
        f"- date: `{payload['date']}`",
        f"- baseline contract: `{payload['baseline_contract']}`",
        f"- source -> target: `{payload['config']['source_model']} -> {payload['config']['target_model']}`",
        f"- calibration file: `{payload['config']['calibration_file']}`",
        f"- seed: `{payload['config']['seed']}`",
        f"- bridge bank size: `{payload['config'].get('bridge_bank_size', 4)}`",
        f"- slice: `{payload['config']['slice_size']}` examples from `{payload['config']['eval_file']}`",
        "",
        "| Base | Residual rank | Bridge bank | Accuracy | Win vs target | Loss vs target | Tie vs target | Numeric coverage | Empty preds | Status | Nonfinite | First bad key | Reused ckpt | Promote? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in payload["rows"]:
        paired = row["paired_vs_target"]
        checks = payload["checks"][row["label"]]
        promote = (
            row.get("status", "ok") == "ok"
            and checks["numeric_extraction_coverage"]
            and checks["beats_target"]
            and checks["row_count_matches_slice"]
            and checks["example_ids_match_target"]
            and checks["no_empty_predictions"]
        )
        first_bad_key = row.get("checkpoint_first_bad_key") or "-"
        lines.append(
            f"| {row['base_label']} | {row['residual_rank']} | {row.get('bridge_bank_size', 4)} | {row['accuracy']:.4f} | "
            f"{paired['win']} | {paired['loss']} | {paired['tie']} | {row['numeric_extraction_coverage']} | "
            f"{row['empty_predictions']} | {row.get('status', 'ok')} | {row.get('checkpoint_nonfinite_numel', 0)} | "
            f"{first_bad_key} | {'yes' if row['reused_existing_checkpoint'] else 'no'} | {'yes' if promote else 'no'} |"
        )
    lines.extend(["", "## Checks", ""])
    for label, checks in payload["checks"].items():
        status = ", ".join(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
        lines.append(f"- `{label}` — {status}")
    lines.extend(["", "## Checkpoint Health", ""])
    for row in payload["rows"]:
        checkpoint_summary = row.get("checkpoint_summary", {})
        top_abs = checkpoint_summary.get("top_abs_tensors", [])
        if top_abs:
            top_entry = top_abs[0]
            top_entry_text = (
                f"{top_entry['key']} (max_abs={top_entry['max_abs']:.4f}, "
                f"nonfinite={top_entry['nonfinite_numel']})"
            )
        else:
            top_entry_text = "-"
        lines.append(
            f"- `{row['label']}` — status={row.get('status', 'ok')}, "
            f"nonfinite_numel={row.get('checkpoint_nonfinite_numel', 0)}, "
            f"first_bad_key={row.get('checkpoint_first_bad_key') or '-'}, "
            f"top_tensor={top_entry_text}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_sweep(config: ResidualSweepConfig) -> dict[str, Any]:
    results_dir = ROOT / config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    materialized = harness.resolve_materialized_eval_file(
        config.materialized_eval_file,
        results_dir=results_dir,
        slice_size=config.slice_size,
    )
    harness.materialize_slice(ROOT / config.eval_file, materialized, config.slice_size)
    baseline_target_records = checkpoint_sweep._load_baseline_target_records(ROOT / config.baseline_results_dir, materialized)

    rows: list[dict[str, Any]] = []
    checks: dict[str, dict[str, bool]] = {}
    for base_label in config.bases:
        for rank in config.ranks:
            checkpoint_path = _checkpoint_path(base_label, rank, config)
            try:
                checkpoint_summary = _calibrate_checkpoint(
                    base_label=base_label,
                    rank=rank,
                    checkpoint_path=checkpoint_path,
                    config=config,
                )
                row, row_checks = _run_candidate(
                    base_label=base_label,
                    rank=rank,
                    checkpoint_path=checkpoint_path,
                    checkpoint_summary=checkpoint_summary,
                    config=config,
                    materialized_eval_file=materialized,
                    baseline_target_records=baseline_target_records,
                    results_dir=results_dir,
                )
            except Exception as exc:
                checkpoint_summary = _safe_checkpoint_summary(checkpoint_path)
                row, row_checks = _failure_row(
                    base_label=base_label,
                    rank=rank,
                    checkpoint_path=checkpoint_path,
                    checkpoint_summary=checkpoint_summary,
                    config=config,
                    error=exc,
                )
            rows.append(row)
            checks[row["label"]] = row_checks

    rows.sort(key=lambda row: (-row["accuracy"], row["base_label"], row["residual_rank"]))
    payload = {
        "date": "2026-04-21",
        "baseline_contract": str(ROOT / config.baseline_results_dir / "gsm8k_smoke_contract_20260421.md"),
        "config": asdict(config),
        "artifacts": {
            "materialized_eval_file": str(materialized),
        },
        "rows": rows,
        "checks": checks,
    }
    smoke._write_json(results_dir / "gsm8k_contract_residual_sweep_20260421.json", payload)
    _write_markdown(results_dir / "gsm8k_contract_residual_sweep_20260421.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen GSM8K32 residual-rank sweep on the live dynalign/tokenbasis bridge families.")
    parser.add_argument("--source-model", default=smoke.DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", default=smoke.DEFAULT_TARGET_MODEL)
    parser.add_argument("--eval-file", default=smoke.DEFAULT_EVAL_FILE)
    parser.add_argument("--slice-size", type=int, default=32)
    parser.add_argument("--materialized-eval-file", default=DEFAULT_MATERIALIZED_EVAL_FILE)
    parser.add_argument("--baseline-results-dir", default=DEFAULT_BASELINE_RESULTS_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_SWEEP_RESULTS_DIR)
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--calibration-file", default=DEFAULT_CALIBRATION_FILE)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gate", type=float, default=0.10)
    parser.add_argument("--kv-transport", default="k_only", choices=["both", "k_only", "v_only"])
    parser.add_argument("--position-selection-ratio", type=float, default=0.5)
    parser.add_argument(
        "--position-selection-metric",
        default="attention",
        choices=["energy", "disagreement", "random", "recency", "attention", "attention_stratified", "query_pool_transport", "attention_disagreement", "attention_disagreement_stratified", "attention_shuffled", "source_attention", "attention_prior"],
    )
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--alignment", default="grouped_subspace_transport")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--bridge-bank-size", type=int, default=4)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--transport-residual-rank", type=int, default=4)
    parser.add_argument("--transport-temperature", type=float, default=0.1)
    parser.add_argument("--transport-sinkhorn-iters", type=int, default=8)
    parser.add_argument("--transport-signature-rank", type=int, default=8)
    parser.add_argument("--transport-signature-weight", type=float, default=0.0)
    parser.add_argument("--whitening", action="store_true")
    parser.add_argument("--target-whitening", action="store_true")
    parser.add_argument("--whitening-streams", choices=["kv", "k", "v"], default="kv")
    parser.add_argument("--target-whitening-streams", choices=["kv", "k", "v"], default="kv")
    parser.add_argument("--conditioning-target-layer", type=int, action="append", dest="conditioning_target_layers")
    parser.add_argument("--fit-ridge-override-lambda", type=float, default=None)
    parser.add_argument("--fit-ridge-override-streams", choices=["kv", "k", "v"], default="kv")
    parser.add_argument("--fit-ridge-override-layer", type=int, action="append", dest="fit_ridge_override_layers")
    parser.add_argument("--fit-ridge-protected-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, action="append", dest="ranks")
    parser.add_argument("--base", action="append", choices=sorted(DEFAULT_BASES), dest="bases")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ResidualSweepConfig(
        source_model=args.source_model,
        target_model=args.target_model,
        eval_file=args.eval_file,
        slice_size=args.slice_size,
        materialized_eval_file=args.materialized_eval_file,
        baseline_results_dir=args.baseline_results_dir,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        calibration_file=args.calibration_file,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        gate=args.gate,
        kv_transport=args.kv_transport,
        position_selection_ratio=args.position_selection_ratio,
        position_selection_metric=args.position_selection_metric,
        source_reasoning_mode=args.source_reasoning_mode,
        use_chat_template=not args.no_chat_template,
        enable_thinking=args.enable_thinking,
        alignment=args.alignment,
        bits=args.bits,
        bridge_bank_size=args.bridge_bank_size,
        ridge_lambda=args.ridge_lambda,
        transport_residual_rank=args.transport_residual_rank,
        transport_temperature=args.transport_temperature,
        transport_sinkhorn_iters=args.transport_sinkhorn_iters,
        transport_signature_rank=args.transport_signature_rank,
        transport_signature_weight=args.transport_signature_weight,
        whitening=args.whitening,
        target_whitening=args.target_whitening,
        whitening_streams=args.whitening_streams,
        target_whitening_streams=args.target_whitening_streams,
        conditioning_target_layers=(
            tuple(args.conditioning_target_layers)
            if args.conditioning_target_layers
            else None
        ),
        fit_ridge_override_lambda=args.fit_ridge_override_lambda,
        fit_ridge_override_streams=args.fit_ridge_override_streams,
        fit_ridge_override_layers=(
            tuple(args.fit_ridge_override_layers)
            if args.fit_ridge_override_layers
            else None
        ),
        fit_ridge_protected_rank=args.fit_ridge_protected_rank,
        seed=args.seed,
        ranks=tuple(args.ranks) if args.ranks else ResidualSweepConfig.ranks,
        bases=tuple(args.bases) if args.bases else ResidualSweepConfig.bases,
    )
    return run_sweep(config)


if __name__ == "__main__":
    main()
