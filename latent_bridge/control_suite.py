from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class CalibrationSpec:
    name: str
    layer_pairing: str
    selection_ratio: float
    seed: int


@dataclass(frozen=True)
class EvalSpec:
    name: str
    methods: tuple[str, ...]
    gate_values: tuple[float, ...]
    quantize: bool
    source_reasoning_mode: str
    include_baselines: bool = False


def default_device() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _coerce_optional_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def parse_summary_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    in_summary = False
    for line in output.splitlines():
        if "=== Summary ===" in line:
            in_summary = True
            continue
        if not in_summary or ":" not in line:
            continue
        name, value = line.split(":", 1)
        try:
            metrics[name.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def default_calibration_specs() -> list[CalibrationSpec]:
    return [
        CalibrationSpec("interp_full_seed0", "interp", 1.0, 0),
        CalibrationSpec("cka_half_seed0", "cka", 0.5, 0),
        CalibrationSpec("cka_quarter_seed0", "cka", 0.25, 0),
        CalibrationSpec("cka_half_seed1", "cka", 0.5, 1),
    ]


def default_eval_specs() -> list[EvalSpec]:
    return [
        EvalSpec(
            name="fused_noquant_plain",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=False,
            source_reasoning_mode="plain",
            include_baselines=True,
        ),
        EvalSpec(
            name="fused_noquant_cot",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=False,
            source_reasoning_mode="cot",
        ),
        EvalSpec(
            name="text_kv_noquant_brief",
            methods=("rotalign_text_kv",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=False,
            source_reasoning_mode="brief_analysis",
        ),
        EvalSpec(
            name="fused_quant_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=True,
            source_reasoning_mode="brief_analysis",
        ),
    ]


def run_logged_command(cmd: list[str], log_path: Path, cwd: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cache_root = cwd / ".hf_home"
    env.setdefault("HF_HOME", str(cache_root))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    env.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hub"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    with log_path.open("a") as log_file:
        log_file.write(f"\n[{iso_now()}] CMD {format_cmd(cmd)}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        chunks: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            log_file.write(line)
            log_file.flush()
        returncode = proc.wait()

    if returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}: {format_cmd(cmd)}")
    return "".join(chunks)


def build_calibrate_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    source_model: str,
    target_model: str,
    calibration_file: str,
    checkpoint_path: Path,
    bits: int,
    rotation: str,
    alignment: str,
    whitening: bool,
    device: str,
    dtype: str,
    spec: CalibrationSpec,
) -> list[str]:
    cmd = [
        python_exe,
        str(repo_root / "scripts" / "calibrate.py"),
        "--source-model",
        source_model,
        "--target-model",
        target_model,
        "--calibration-file",
        calibration_file,
        "--output",
        str(checkpoint_path),
        "--bits",
        str(bits),
        "--rotation",
        rotation,
        "--alignment",
        alignment,
        "--layer-pairing",
        spec.layer_pairing,
        "--layer-selection-ratio",
        str(spec.selection_ratio),
        "--seed",
        str(spec.seed),
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    if whitening:
        cmd.append("--whitening")
    return cmd


def build_evaluate_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    source_model: str,
    target_model: str,
    eval_file: str,
    checkpoint_path: Path,
    task_type: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    spec: EvalSpec,
) -> list[str]:
    methods = list(spec.methods)
    if spec.include_baselines:
        methods = ["target", "t2t", *methods]
    cmd = [
        python_exe,
        str(repo_root / "scripts" / "evaluate.py"),
        "--translator",
        str(checkpoint_path),
        "--source-model",
        source_model,
        "--target-model",
        target_model,
        "--eval-file",
        eval_file,
        "--task-type",
        task_type,
        "--device",
        device,
        "--dtype",
        dtype,
        "--max-new-tokens",
        str(max_new_tokens),
        "--source-reasoning-mode",
        spec.source_reasoning_mode,
        "--methods",
        *methods,
        "--gate-mode",
        "sweep",
        "--gate-values",
        *[str(v) for v in spec.gate_values],
    ]
    if not spec.quantize:
        cmd.append("--no-quantize")
    return cmd


def write_summary(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "suite_results.jsonl"
    csv_path = out_dir / "suite_results.csv"
    md_path = out_dir / "latest_summary.md"

    with jsonl_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    fieldnames = [
        "checkpoint_tag",
        "eval_name",
        "layer_pairing",
        "selection_ratio",
        "seed",
        "source_reasoning_mode",
        "quantize",
        "target_alone",
        "text_to_text",
        "best_metric",
        "best_value",
        "best_minus_target",
        "best_minus_text_to_text",
        "best_bytes",
        "best_ttft_sec",
        "best_tokens_per_sec",
        "log_file",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "checkpoint_tag": record["checkpoint_tag"],
                    "eval_name": record["eval_name"],
                    "layer_pairing": record["layer_pairing"],
                    "selection_ratio": record["selection_ratio"],
                    "seed": record["seed"],
                    "source_reasoning_mode": record["source_reasoning_mode"],
                    "quantize": record["quantize"],
                    "target_alone": record.get("target_alone"),
                    "text_to_text": record.get("text_to_text"),
                    "best_metric": record["best_metric"],
                    "best_value": record["best_value"],
                    "best_minus_target": record.get("best_minus_target"),
                    "best_minus_text_to_text": record.get("best_minus_text_to_text"),
                    "best_bytes": record.get("best_bytes"),
                    "best_ttft_sec": record.get("best_ttft_sec"),
                    "best_tokens_per_sec": record.get("best_tokens_per_sec"),
                    "log_file": record["log_file"],
                }
            )

    lines = [
        "# RotAlign Control Suite",
        "",
        "| Checkpoint | Eval | Target | T2T | Best Metric | Value | Δ vs Target | Δ vs T2T | Bytes | TTFT (s) | Tok/s |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(records, key=lambda r: float(r["best_value"]), reverse=True):
        lines.append(
            "| "
            f"{record['checkpoint_tag']} | "
            f"{record['eval_name']} | "
            f"{_coerce_optional_float(record.get('target_alone')):.4f} | "
            f"{_coerce_optional_float(record.get('text_to_text')):.4f} | "
            f"{record['best_metric']} | "
            f"{record['best_value']:.4f} | "
            f"{_coerce_optional_float(record.get('best_minus_target')):.4f} | "
            f"{_coerce_optional_float(record.get('best_minus_text_to_text')):.4f} | "
            f"{_coerce_optional_float(record.get('best_bytes')):.1f} | "
            f"{_coerce_optional_float(record.get('best_ttft_sec')):.4f} | "
            f"{_coerce_optional_float(record.get('best_tokens_per_sec')):.3f} |"
        )
    if not records:
        lines.append("| pending | pending |  |  | pending |  |  |  |  |  |  |")
    md_path.write_text("\n".join(lines) + "\n")


def best_metric_for_eval(
    metrics: dict[str, float],
    methods: tuple[str, ...],
    gate_values: tuple[float, ...],
    *,
    include_baselines: bool = False,
) -> tuple[str, float]:
    prefixes: list[str] = []
    for method in methods:
        if method == "rotalign":
            prefixes.append("rotalign_kv_gate_")
        elif method == "rotalign_text_kv":
            prefixes.append("rotalign_text_kv_hybrid_gate_")
        elif method == "rotalign_translated":
            prefixes.append("rotalign_translated_only_gate_")
        elif method == "t2t":
            prefixes.append("text_to_text")
        else:
            prefixes.append(method)
    if include_baselines:
        prefixes.extend(["target_alone", "text_to_text"])
    candidates: list[tuple[str, float]] = []
    for key, value in metrics.items():
        if any(key.startswith(prefix) and not key.endswith(("_bits", "_bytes", "_ttft_sec", "_tokens_per_sec", "_examples_per_sec", "_latency_sec", "_generated_tokens_avg")) for prefix in prefixes):
            candidates.append((key, value))
    if not candidates:
        return "none", float("nan")
    return max(candidates, key=lambda item: item[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--results-dir", default="results/overnight_control_suite")
    parser.add_argument("--checkpoint-dir", default="checkpoints/overnight_control_suite")
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--rotation", default="orthogonal", choices=["identity", "orthogonal", "hadamard"])
    parser.add_argument("--alignment", default="auto", choices=["auto", "identity", "procrustes", "procrustes_rand", "ridge", "cca", "reduced_rank"])
    parser.set_defaults(whitening=True)
    parser.add_argument("--no-whitening", dest="whitening", action="store_false")
    parser.add_argument("--task-type", default="generation", choices=["auto", "mcq", "generation"])
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable
    results_dir = Path(args.results_dir)
    logs_dir = results_dir / "logs"
    checkpoints_dir = Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    calibration_specs = default_calibration_specs()
    eval_specs = default_eval_specs()

    plan = {
        "source_model": args.source_model,
        "target_model": args.target_model,
        "calibration_file": args.calibration_file,
        "eval_file": args.eval_file,
        "rotation": args.rotation,
        "alignment": args.alignment,
        "whitening": args.whitening,
        "bits": args.bits,
        "task_type": args.task_type,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "calibration_specs": [asdict(spec) for spec in calibration_specs],
        "eval_specs": [asdict(spec) for spec in eval_specs],
    }
    (results_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    if args.dry_run:
        for spec in calibration_specs:
            ckpt = checkpoints_dir / f"{spec.name}.pt"
            print(format_cmd(build_calibrate_cmd(
                python_exe=python_exe,
                repo_root=repo_root,
                source_model=args.source_model,
                target_model=args.target_model,
                calibration_file=args.calibration_file,
                checkpoint_path=ckpt,
                bits=args.bits,
                rotation=args.rotation,
                alignment=args.alignment,
                whitening=args.whitening,
                device=args.device,
                dtype=args.dtype,
                spec=spec,
            )))
            for eval_spec in eval_specs:
                print(format_cmd(build_evaluate_cmd(
                    python_exe=python_exe,
                    repo_root=repo_root,
                    source_model=args.source_model,
                    target_model=args.target_model,
                    eval_file=args.eval_file,
                    checkpoint_path=ckpt,
                    task_type=args.task_type,
                    device=args.device,
                    dtype=args.dtype,
                    max_new_tokens=args.max_new_tokens,
                    spec=eval_spec,
                )))
        return

    started = time.monotonic()
    deadline = started + args.budget_hours * 60 * 60
    records: list[dict[str, Any]] = []

    for spec in calibration_specs:
        if time.monotonic() >= deadline:
            break
        checkpoint_path = checkpoints_dir / f"{spec.name}.pt"
        calibrate_log = logs_dir / f"{spec.name}_calibrate.log"
        calibrate_cmd = build_calibrate_cmd(
            python_exe=python_exe,
            repo_root=repo_root,
            source_model=args.source_model,
            target_model=args.target_model,
            calibration_file=args.calibration_file,
            checkpoint_path=checkpoint_path,
            bits=args.bits,
            rotation=args.rotation,
            alignment=args.alignment,
            whitening=args.whitening,
            device=args.device,
            dtype=args.dtype,
            spec=spec,
        )
        run_logged_command(calibrate_cmd, calibrate_log, repo_root)

        for eval_spec in eval_specs:
            if time.monotonic() >= deadline:
                break
            eval_log = logs_dir / f"{spec.name}_{eval_spec.name}.log"
            eval_cmd = build_evaluate_cmd(
                python_exe=python_exe,
                repo_root=repo_root,
                source_model=args.source_model,
                target_model=args.target_model,
                eval_file=args.eval_file,
                checkpoint_path=checkpoint_path,
                task_type=args.task_type,
                device=args.device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                spec=eval_spec,
            )
            output = run_logged_command(eval_cmd, eval_log, repo_root)
            metrics = parse_summary_metrics(output)
            best_metric, best_value = best_metric_for_eval(
                metrics,
                eval_spec.methods,
                eval_spec.gate_values,
                include_baselines=eval_spec.include_baselines,
            )
            target_alone = metrics.get("target_alone")
            text_to_text = metrics.get("text_to_text")
            best_bytes = metrics.get(f"{best_metric}_bytes") if best_metric != "none" else None
            best_ttft = metrics.get(f"{best_metric}_ttft_sec") if best_metric != "none" else None
            best_toks = metrics.get(f"{best_metric}_tokens_per_sec") if best_metric != "none" else None
            record = {
                "timestamp": iso_now(),
                "checkpoint_tag": spec.name,
                "layer_pairing": spec.layer_pairing,
                "selection_ratio": spec.selection_ratio,
                "seed": spec.seed,
                "eval_name": eval_spec.name,
                "source_reasoning_mode": eval_spec.source_reasoning_mode,
                "quantize": eval_spec.quantize,
                "methods": list(eval_spec.methods),
                "gate_values": list(eval_spec.gate_values),
                "target_alone": target_alone,
                "text_to_text": text_to_text,
                "best_metric": best_metric,
                "best_value": best_value,
                "best_minus_target": None if target_alone is None else best_value - target_alone,
                "best_minus_text_to_text": None if text_to_text is None else best_value - text_to_text,
                "best_bytes": best_bytes,
                "best_ttft_sec": best_ttft,
                "best_tokens_per_sec": best_toks,
                "checkpoint_path": str(checkpoint_path),
                "log_file": str(eval_log),
                "metrics": metrics,
            }
            records.append(record)
            write_summary(records, results_dir)


if __name__ == "__main__":
    main()
