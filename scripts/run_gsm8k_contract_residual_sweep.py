#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gsm8k_contract_checkpoint_sweep as checkpoint_sweep
from scripts import run_gsm8k_smoke_contract as smoke


DEFAULT_BASELINE_RESULTS_DIR = "results/gsm8k_smoke_contract_20260421"
DEFAULT_SWEEP_RESULTS_DIR = "results/gsm8k_contract_residual_sweep_20260421"
DEFAULT_CHECKPOINTS_DIR = "checkpoints/gsm8k_contract_residual_sweep_20260421"
DEFAULT_MATERIALIZED_EVAL_FILE = "/tmp/gsm8k_eval_32.jsonl"
DEFAULT_CALIBRATION_FILE = ".debug/calibration_64.txt"

DEFAULT_BASES: dict[str, dict[str, str | int]] = {
    "dynalign_module_replace": {
        "quantization_correction": "bridge_ridge_qk_dynalign_module_replace",
        "checkpoint_path": "checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt",
        "existing_rank": 8,
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
    materialized_eval_file: str = DEFAULT_MATERIALIZED_EVAL_FILE
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
    ridge_lambda: float = 1e-3
    transport_residual_rank: int = 4
    transport_temperature: float = 0.1
    transport_sinkhorn_iters: int = 8
    transport_signature_rank: int = 8
    transport_signature_weight: float = 0.0
    ranks: tuple[int, ...] = (2, 4, 8, 16)
    bases: tuple[str, ...] = ("dynalign_module_replace", "tokenbasis_replace")


def _run(cmd: list[str]) -> None:
    smoke._run(cmd, cwd=ROOT)


def _candidate_label(base_label: str, rank: int) -> str:
    return f"{base_label}_residrank{rank}"


def _checkpoint_path(base_label: str, rank: int, config: ResidualSweepConfig) -> pathlib.Path:
    base_meta = DEFAULT_BASES[base_label]
    if rank == int(base_meta["existing_rank"]):
        return ROOT / str(base_meta["checkpoint_path"])
    filename = (
        f"qwen25_to_qwen3_grouped_subspace_transport_w010_r{rank}_"
        f"{base_label}_cal64_chat.pt"
    )
    return ROOT / config.checkpoints_dir / base_label / filename


def _calibrate_checkpoint(
    *,
    base_label: str,
    rank: int,
    checkpoint_path: pathlib.Path,
    config: ResidualSweepConfig,
) -> None:
    if checkpoint_path.exists():
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    base_meta = DEFAULT_BASES[base_label]
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
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
        "--source-reasoning-mode",
        config.source_reasoning_mode,
        "--device",
        config.device,
        "--dtype",
        config.dtype,
        "--seed",
        "0",
    ]
    if config.use_chat_template:
        cmd.extend(
            [
                "--source-use-chat-template",
                "--target-use-chat-template",
                "--source-enable-thinking",
                "false" if not config.enable_thinking else "true",
                "--target-enable-thinking",
                "false" if not config.enable_thinking else "true",
            ]
        )
    _run(cmd)


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
    config: ResidualSweepConfig,
    materialized_eval_file: pathlib.Path,
    baseline_target_records: list[dict[str, Any]],
    results_dir: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, bool]]:
    label = _candidate_label(base_label, rank)
    prediction_output = results_dir / f"{label}.jsonl"
    if not prediction_output.exists():
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
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
        ]
        if config.use_chat_template:
            cmd.extend(
                [
                    "--source-use-chat-template",
                    "--target-use-chat-template",
                    "--source-enable-thinking",
                    "false" if not config.enable_thinking else "true",
                    "--target-enable-thinking",
                    "false" if not config.enable_thinking else "true",
                ]
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
    row["reused_existing_checkpoint"] = rank == int(DEFAULT_BASES[base_label]["existing_rank"])
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
        f"- slice: `{payload['config']['slice_size']}` examples from `{payload['config']['eval_file']}`",
        "",
        "| Base | Residual rank | Accuracy | Win vs target | Loss vs target | Tie vs target | Numeric coverage | Empty preds | Reused ckpt | Promote? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        paired = row["paired_vs_target"]
        checks = payload["checks"][row["label"]]
        promote = (
            checks["numeric_extraction_coverage"]
            and checks["beats_target"]
            and checks["row_count_matches_slice"]
            and checks["example_ids_match_target"]
            and checks["no_empty_predictions"]
        )
        lines.append(
            f"| {row['base_label']} | {row['residual_rank']} | {row['accuracy']:.4f} | "
            f"{paired['win']} | {paired['loss']} | {paired['tie']} | {row['numeric_extraction_coverage']} | "
            f"{row['empty_predictions']} | {'yes' if row['reused_existing_checkpoint'] else 'no'} | {'yes' if promote else 'no'} |"
        )
    lines.extend(["", "## Checks", ""])
    for label, checks in payload["checks"].items():
        status = ", ".join(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
        lines.append(f"- `{label}` — {status}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_sweep(config: ResidualSweepConfig) -> dict[str, Any]:
    results_dir = ROOT / config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    materialized = pathlib.Path(config.materialized_eval_file)
    smoke._materialize_slice(ROOT / config.eval_file, materialized, config.slice_size)
    baseline_target_records = checkpoint_sweep._load_baseline_target_records(ROOT / config.baseline_results_dir, materialized)

    rows: list[dict[str, Any]] = []
    checks: dict[str, dict[str, bool]] = {}
    for base_label in config.bases:
        for rank in config.ranks:
            checkpoint_path = _checkpoint_path(base_label, rank, config)
            _calibrate_checkpoint(
                base_label=base_label,
                rank=rank,
                checkpoint_path=checkpoint_path,
                config=config,
            )
            row, row_checks = _run_candidate(
                base_label=base_label,
                rank=rank,
                checkpoint_path=checkpoint_path,
                config=config,
                materialized_eval_file=materialized,
                baseline_target_records=baseline_target_records,
                results_dir=results_dir,
            )
            rows.append(row)
            checks[row["label"]] = row_checks

    rows.sort(key=lambda row: (-row["accuracy"], row["base_label"], row["residual_rank"]))
    payload = {
        "date": "2026-04-21",
        "baseline_contract": str(ROOT / config.baseline_results_dir / "gsm8k_smoke_contract_20260421.md"),
        "config": asdict(config),
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
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--transport-residual-rank", type=int, default=4)
    parser.add_argument("--transport-temperature", type=float, default=0.1)
    parser.add_argument("--transport-sinkhorn-iters", type=int, default=8)
    parser.add_argument("--transport-signature-rank", type=int, default=8)
    parser.add_argument("--transport-signature-weight", type=float, default=0.0)
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
        ridge_lambda=args.ridge_lambda,
        transport_residual_rank=args.transport_residual_rank,
        transport_temperature=args.transport_temperature,
        transport_sinkhorn_iters=args.transport_sinkhorn_iters,
        transport_signature_rank=args.transport_signature_rank,
        transport_signature_weight=args.transport_signature_weight,
        ranks=tuple(args.ranks) if args.ranks else ResidualSweepConfig.ranks,
        bases=tuple(args.bases) if args.bases else ResidualSweepConfig.bases,
    )
    return run_sweep(config)


if __name__ == "__main__":
    main()
