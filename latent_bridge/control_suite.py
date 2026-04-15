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
from typing import Any, TypeVar


DEFAULT_SOURCE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-0.6B"
NamedSpecT = TypeVar("NamedSpecT")


@dataclass(frozen=True)
class CalibrationSpec:
    name: str
    layer_pairing: str
    selection_ratio: float
    seed: int
    rotation: str | None = None
    alignment: str | None = None
    whitening: bool | None = None
    bits: int | None = None
    alignment_rank: int | None = None


@dataclass(frozen=True)
class EvalSpec:
    name: str
    methods: tuple[str, ...]
    gate_values: tuple[float, ...]
    quantize: bool
    source_reasoning_mode: str
    include_baselines: bool = False
    source_kv_control: str = "real"
    quantization_control: str = "real"
    translated_kv_control: str = "real"


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


def load_existing_records(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def default_calibration_specs() -> list[CalibrationSpec]:
    return [
        CalibrationSpec("interp_full_seed0", "interp", 1.0, 0),
        CalibrationSpec("cka_half_seed0", "cka", 0.5, 0),
        CalibrationSpec("cka_quarter_seed0", "cka", 0.25, 0),
        CalibrationSpec("cka_half_seed1", "cka", 0.5, 1),
        CalibrationSpec("no_rotation_cka_half_seed1", "cka", 0.5, 1, rotation="identity"),
        CalibrationSpec("hadamard_cka_half_seed1", "cka", 0.5, 1, rotation="hadamard"),
        CalibrationSpec("dct_cka_half_seed1", "cka", 0.5, 1, rotation="dct"),
        CalibrationSpec("identity_align_cka_half_seed1", "cka", 0.5, 1, alignment="identity"),
        CalibrationSpec("shifted_layers_half_seed1", "shifted", 0.5, 1),
        CalibrationSpec("random_layers_half_seed1", "random", 0.5, 1),
        CalibrationSpec("lowrank128_cka_half_seed1", "cka", 0.5, 1, alignment="reduced_rank", alignment_rank=128),
        CalibrationSpec("bits2_cka_half_seed1", "cka", 0.5, 1, bits=2),
        CalibrationSpec("bits3_cka_half_seed1", "cka", 0.5, 1, bits=3),
        CalibrationSpec("bits6_cka_half_seed1", "cka", 0.5, 1, bits=6),
        CalibrationSpec("bits8_cka_half_seed1", "cka", 0.5, 1, bits=8),
    ]


def default_eval_specs() -> list[EvalSpec]:
    return [
        EvalSpec(
            name="baseline_plain",
            methods=(),
            gate_values=(),
            quantize=False,
            source_reasoning_mode="plain",
            include_baselines=True,
        ),
        EvalSpec(
            name="baseline_brief",
            methods=(),
            gate_values=(),
            quantize=False,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
        ),
        EvalSpec(
            name="baseline_cot",
            methods=(),
            gate_values=(),
            quantize=False,
            source_reasoning_mode="cot",
            include_baselines=True,
        ),
        EvalSpec(
            name="fused_noquant_plain",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=False,
            source_reasoning_mode="plain",
            include_baselines=True,
        ),
        EvalSpec(
            name="fused_noquant_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=False,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
        ),
        EvalSpec(
            name="fused_noquant_cot",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=False,
            source_reasoning_mode="cot",
            include_baselines=True,
        ),
        EvalSpec(
            name="translated_noquant_brief",
            methods=("rotalign_translated",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=False,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
        ),
        EvalSpec(
            name="text_kv_noquant_brief",
            methods=("rotalign_text_kv",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=False,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
        ),
        EvalSpec(
            name="fused_quant_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=True,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
        ),
        EvalSpec(
            name="fused_quant_noise_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=True,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
            quantization_control="matched_noise",
        ),
        EvalSpec(
            name="fused_quant_random_kv_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=True,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
            source_kv_control="random",
        ),
        EvalSpec(
            name="fused_quant_shuffle_kv_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=True,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
            source_kv_control="shuffle_positions",
        ),
        EvalSpec(
            name="fused_quant_zero_kv_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=True,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
            source_kv_control="zero",
        ),
        EvalSpec(
            name="fused_quant_zero_translated_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=True,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
            translated_kv_control="zero",
        ),
        EvalSpec(
            name="fused_quant_random_translated_brief",
            methods=("rotalign",),
            gate_values=(0.15, 0.25, 0.30),
            quantize=True,
            source_reasoning_mode="brief_analysis",
            include_baselines=True,
            translated_kv_control="random",
        ),
    ]


def _filter_named_specs(
    specs: list[NamedSpecT], selected_names: list[str] | None
) -> list[NamedSpecT]:
    if not selected_names:
        return specs
    wanted = list(dict.fromkeys(selected_names))
    by_name = {getattr(spec, "name"): spec for spec in specs}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"Unknown spec name(s): {', '.join(missing)}")
    return [by_name[name] for name in wanted]


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
    effective_bits = spec.bits if spec.bits is not None else bits
    effective_rotation = spec.rotation if spec.rotation is not None else rotation
    effective_alignment = spec.alignment if spec.alignment is not None else alignment
    effective_whitening = spec.whitening if spec.whitening is not None else whitening
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
        str(effective_bits),
        "--rotation",
        effective_rotation,
        "--alignment",
        effective_alignment,
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
    if spec.alignment_rank is not None:
        cmd.extend(["--alignment-rank", str(spec.alignment_rank)])
    if effective_whitening:
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
    gate_search_file: str | None,
    gate_search_limit: int,
    spec: EvalSpec,
    prediction_output: Path | None = None,
) -> list[str]:
    methods = list(spec.methods)
    if spec.include_baselines:
        methods = ["target", "t2t", *methods]
    uses_rotalign = any(
        method in {"rotalign", "rotalign_translated", "rotalign_fused", "rotalign_text_kv"}
        for method in methods
    )
    uses_gate_dependent_rotalign = any(
        method in {"rotalign", "rotalign_fused", "rotalign_text_kv"}
        for method in methods
    )
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
    ]
    if uses_gate_dependent_rotalign and gate_search_file:
        cmd.extend(
            [
                "--gate-mode",
                "search",
                "--gate-search-file",
                gate_search_file,
                "--gate-search-limit",
                str(gate_search_limit),
            ]
        )
    elif uses_gate_dependent_rotalign:
        cmd.extend(["--gate-mode", "sweep"])
    else:
        cmd.extend(["--gate-mode", "checkpoint"])
    if uses_gate_dependent_rotalign and spec.gate_values:
        cmd.extend(["--gate-values", *[str(v) for v in spec.gate_values]])
    if uses_rotalign and spec.source_kv_control != "real":
        cmd.extend(["--source-kv-control", spec.source_kv_control])
    if uses_rotalign and spec.quantization_control != "real":
        cmd.extend(["--quantization-control", spec.quantization_control])
    if uses_rotalign and spec.translated_kv_control != "real":
        cmd.extend(["--translated-kv-control", spec.translated_kv_control])
    if not spec.quantize:
        cmd.append("--no-quantize")
    if prediction_output is not None:
        cmd.extend(["--prediction-output", str(prediction_output)])
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
        "source_kv_control",
        "quantization_control",
        "translated_kv_control",
        "quantize",
        "target_alone",
        "text_to_text",
        "best_metric",
        "best_value",
        "best_minus_target",
        "best_minus_text_to_text",
        "best_bytes",
        "latent_metric",
        "latent_value",
        "latent_minus_target",
        "latent_minus_text_to_text",
        "latent_bytes",
        "latent_bits",
        "latent_latency_sec",
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
                    "source_kv_control": record["source_kv_control"],
                    "quantization_control": record.get("quantization_control", "real"),
                    "translated_kv_control": record.get("translated_kv_control", "real"),
                    "quantize": record["quantize"],
                    "target_alone": record.get("target_alone"),
                    "text_to_text": record.get("text_to_text"),
                    "best_metric": record["best_metric"],
                    "best_value": record["best_value"],
                    "best_minus_target": record.get("best_minus_target"),
                    "best_minus_text_to_text": record.get("best_minus_text_to_text"),
                    "best_bytes": record.get("best_bytes"),
                    "latent_metric": record.get("latent_metric"),
                    "latent_value": record.get("latent_value"),
                    "latent_minus_target": record.get("latent_minus_target"),
                    "latent_minus_text_to_text": record.get("latent_minus_text_to_text"),
                    "latent_bytes": record.get("latent_bytes"),
                    "latent_bits": record.get("latent_bits"),
                    "latent_latency_sec": record.get("latent_latency_sec"),
                    "best_ttft_sec": record.get("best_ttft_sec"),
                    "best_tokens_per_sec": record.get("best_tokens_per_sec"),
                    "log_file": record["log_file"],
                }
            )

    lines = [
        "# RotAlign Control Suite",
        "",
        "| Checkpoint | Eval | Target | T2T | Best Metric | Value | Δ vs Target | Latent Metric | Latent | Latent ΔT | Latent Bytes | Latency (s) |",
        "|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|",
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
            f"{record.get('latent_metric') or 'none'} | "
            f"{_coerce_optional_float(record.get('latent_value')):.4f} | "
            f"{_coerce_optional_float(record.get('latent_minus_target')):.4f} | "
            f"{_coerce_optional_float(record.get('latent_bytes')):.1f} | "
            f"{_coerce_optional_float(record.get('latent_latency_sec')):.4f} |"
        )
    if not records:
        lines.append("| pending | pending |  |  | pending |  |  | pending |  |  |  |  |")
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
            prefixes.extend(["rotalign_kv_gate_", "rotalign_kv"])
        elif method == "rotalign_fused":
            prefixes.extend(["rotalign_fused_gate_", "rotalign_fused"])
        elif method == "rotalign_text_kv":
            prefixes.extend(["rotalign_text_kv_hybrid_gate_", "rotalign_text_kv_hybrid"])
        elif method == "rotalign_translated":
            prefixes.extend(["rotalign_translated_only_gate_", "rotalign_translated_only"])
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


def metric_sidecar(metrics: dict[str, float], metric: str) -> dict[str, float | None]:
    return {
        "bits": metrics.get(f"{metric}_bits"),
        "bytes": metrics.get(f"{metric}_bytes"),
        "latency_sec": metrics.get(f"{metric}_latency_sec"),
        "ttft_sec": metrics.get(f"{metric}_ttft_sec"),
        "tokens_per_sec": metrics.get(f"{metric}_tokens_per_sec"),
    }


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
    parser.add_argument("--rotation", default="orthogonal", choices=["identity", "orthogonal", "hadamard", "dct"])
    parser.add_argument("--alignment", default="auto", choices=["auto", "identity", "procrustes", "procrustes_rand", "ridge", "cca", "reduced_rank"])
    parser.set_defaults(whitening=True)
    parser.add_argument("--no-whitening", dest="whitening", action="store_false")
    parser.add_argument("--task-type", default="generation", choices=["auto", "mcq", "generation"])
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--calibration-specs",
        nargs="+",
        default=None,
        help="Optional list of calibration spec names to run.",
    )
    parser.add_argument(
        "--eval-specs",
        nargs="+",
        default=None,
        help="Optional list of eval spec names to run.",
    )
    parser.add_argument(
        "--gate-search-file",
        default=None,
        help="Optional held-out JSONL split used for per-layer gate search before final evaluation.",
    )
    parser.add_argument(
        "--gate-search-limit",
        type=int,
        default=30,
        help="Maximum number of held-out examples to use during gate search.",
    )
    parser.set_defaults(reuse_checkpoints=True)
    parser.add_argument(
        "--no-reuse-checkpoints",
        dest="reuse_checkpoints",
        action="store_false",
        help="Always recalibrate even if a checkpoint already exists.",
    )
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

    calibration_specs = _filter_named_specs(default_calibration_specs(), args.calibration_specs)
    eval_specs = _filter_named_specs(default_eval_specs(), args.eval_specs)

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
        "gate_search_file": args.gate_search_file,
        "gate_search_limit": args.gate_search_limit,
        "reuse_checkpoints": args.reuse_checkpoints,
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
                    gate_search_file=args.gate_search_file,
                    gate_search_limit=args.gate_search_limit,
                    spec=eval_spec,
                    prediction_output=results_dir / "predictions" / f"{spec.name}_{eval_spec.name}.jsonl",
                )))
        return

    started = time.monotonic()
    deadline = started + args.budget_hours * 60 * 60
    records = load_existing_records(results_dir / "suite_results.jsonl")
    completed = {
        (record.get("checkpoint_tag"), record.get("eval_name"))
        for record in records
    }

    for spec in calibration_specs:
        if time.monotonic() >= deadline:
            break
        checkpoint_path = checkpoints_dir / f"{spec.name}.pt"
        calibrate_log = logs_dir / f"{spec.name}_calibrate.log"
        if not (args.reuse_checkpoints and checkpoint_path.exists() and checkpoint_path.stat().st_size > 0):
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
            record_key = (spec.name, eval_spec.name)
            if record_key in completed:
                continue
            eval_log = logs_dir / f"{spec.name}_{eval_spec.name}.log"
            prediction_output = results_dir / "predictions" / f"{spec.name}_{eval_spec.name}.jsonl"
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
                gate_search_file=args.gate_search_file,
                gate_search_limit=args.gate_search_limit,
                spec=eval_spec,
                prediction_output=prediction_output,
            )
            output = run_logged_command(eval_cmd, eval_log, repo_root)
            metrics = parse_summary_metrics(output)
            best_metric, best_value = best_metric_for_eval(
                metrics,
                eval_spec.methods,
                eval_spec.gate_values,
                include_baselines=eval_spec.include_baselines,
            )
            latent_metric, latent_value = best_metric_for_eval(
                metrics,
                eval_spec.methods,
                eval_spec.gate_values,
                include_baselines=False,
            )
            target_alone = metrics.get("target_alone")
            text_to_text = metrics.get("text_to_text")
            best_bytes = metrics.get(f"{best_metric}_bytes") if best_metric != "none" else None
            best_ttft = metrics.get(f"{best_metric}_ttft_sec") if best_metric != "none" else None
            best_toks = metrics.get(f"{best_metric}_tokens_per_sec") if best_metric != "none" else None
            latent_sidecar = metric_sidecar(metrics, latent_metric) if latent_metric != "none" else {}
            record = {
                "timestamp": iso_now(),
                "checkpoint_tag": spec.name,
                "layer_pairing": spec.layer_pairing,
                "selection_ratio": spec.selection_ratio,
                "seed": spec.seed,
                "eval_name": eval_spec.name,
                "source_reasoning_mode": eval_spec.source_reasoning_mode,
                "source_kv_control": eval_spec.source_kv_control,
                "quantization_control": eval_spec.quantization_control,
                "translated_kv_control": eval_spec.translated_kv_control,
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
                "latent_metric": latent_metric,
                "latent_value": None if latent_metric == "none" else latent_value,
                "latent_minus_target": None if target_alone is None or latent_metric == "none" else latent_value - target_alone,
                "latent_minus_text_to_text": None if text_to_text is None or latent_metric == "none" else latent_value - text_to_text,
                "latent_bits": latent_sidecar.get("bits"),
                "latent_bytes": latent_sidecar.get("bytes"),
                "latent_latency_sec": latent_sidecar.get("latency_sec"),
                "latent_ttft_sec": latent_sidecar.get("ttft_sec"),
                "latent_tokens_per_sec": latent_sidecar.get("tokens_per_sec"),
                "checkpoint_path": str(checkpoint_path),
                "log_file": str(eval_log),
                "prediction_file": str(prediction_output),
                "metrics": metrics,
            }
            records.append(record)
            completed.add(record_key)
            write_summary(records, results_dir)


if __name__ == "__main__":
    main()
