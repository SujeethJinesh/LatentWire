#!/usr/bin/env python3
"""Phased overnight runner for small-model RotAlign pilots.

This script orchestrates the existing repo-root RotAlign CLI entrypoints:

- ``scripts/calibrate.py``
- ``scripts/evaluate.py``
- ``scripts/ablation_sweep.py``

It is intended for unattended overnight use on one calibration file and one
generation eval file. The default schedule is restricted to pairs that are
supported by the current KV-only runner:

1. Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B
2. Qwen/Qwen2.5-0.5B-Instruct -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

For each pair it writes incremental JSONL status rows, captures a per-pair log,
retries failing subprocesses once, and records the parsed evaluation summary.
If enough budget remains after the pilots, it runs a tiny 4-config ablation
on the best pair.
"""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
]


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
    if "rotalign_kv" not in metrics:
        for key, value in metrics.items():
            if key.startswith("rotalign_kv_gate_"):
                metrics["rotalign_kv"] = value
                break
    return metrics


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


@dataclass
class StepResult:
    returncode: int
    elapsed_sec: float
    output: str


class StatusWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": iso_now(),
            "event": event,
            **fields,
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        summary = fields.get("pair_tag") or fields.get("phase") or "batch"
        print(f"[{record['timestamp']}] {event} :: {summary}", flush=True)


def write_pair_summaries(pair_records: list[dict[str, Any]], primary_metric: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "pair_results.jsonl"
    csv_path = out_dir / "pair_results.csv"
    md_path = out_dir / "latest_summary.md"

    with jsonl_path.open("w") as handle:
        for record in pair_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    fieldnames = [
        "pair_index",
        "pair_tag",
        "source_model",
        "target_model",
        "elapsed_sec",
        "target_alone",
        "text_to_text",
        "rotalign_kv",
        "checkpoint_path",
        "log_file",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in pair_records:
            metrics = record.get("metrics", {})
            writer.writerow(
                {
                    "pair_index": record.get("pair_index"),
                    "pair_tag": record.get("pair_tag"),
                    "source_model": record.get("source_model"),
                    "target_model": record.get("target_model"),
                    "elapsed_sec": record.get("elapsed_sec"),
                    "target_alone": metrics.get("target_alone"),
                    "text_to_text": metrics.get("text_to_text"),
                    "rotalign_kv": metrics.get("rotalign_kv"),
                    "checkpoint_path": record.get("checkpoint_path"),
                    "log_file": record.get("log_file"),
                }
            )

    lines = [
        "# Overnight RotAlign Status",
        "",
        f"Primary metric: `{primary_metric}`",
        "",
        "| Pair | Target | T2T | RotAlign | Elapsed (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    sorted_records = sorted(
        pair_records,
        key=lambda record: float(record.get("metrics", {}).get(primary_metric, float("-inf"))),
        reverse=True,
    )
    for record in sorted_records:
        metrics = record.get("metrics", {})
        lines.append(
            "| "
            f"{record.get('pair_tag')} | "
            f"{metrics.get('target_alone', float('nan')):.4f} | "
            f"{metrics.get('text_to_text', float('nan')):.4f} | "
            f"{metrics.get('rotalign_kv', float('nan')):.4f} | "
            f"{record.get('elapsed_sec', float('nan')):.1f} |"
        )
    if not pair_records:
        lines.append("| pending |  |  |  |  |")
    md_path.write_text("\n".join(lines) + "\n")


def run_logged_command(cmd: list[str], log_path: Path, cwd: Path) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
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

    return StepResult(
        returncode=returncode,
        elapsed_sec=time.monotonic() - started,
        output="".join(chunks),
    )


def run_with_retries(
    *,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    status: StatusWriter,
    phase: str,
    pair_tag: str,
    max_attempts: int,
    retry_backoff_sec: int,
    extra_status: dict[str, Any],
) -> StepResult:
    merged_status = dict(extra_status)
    merged_status.pop("log_file", None)
    for attempt in range(1, max_attempts + 1):
        status.emit(
            f"{phase}_started",
            phase=phase,
            pair_tag=pair_tag,
            attempt=attempt,
            command=cmd,
            log_file=str(log_path),
            **merged_status,
        )
        try:
            result = run_logged_command(cmd, log_path=log_path, cwd=cwd)
        except Exception as exc:
            will_retry = attempt < max_attempts
            status.emit(
                f"{phase}_failed",
                phase=phase,
                pair_tag=pair_tag,
                attempt=attempt,
                returncode=None,
                will_retry=will_retry,
                log_file=str(log_path),
                error=str(exc),
                **merged_status,
            )
            if will_retry:
                time.sleep(retry_backoff_sec * attempt)
                continue
            raise
        if result.returncode == 0:
            status.emit(
                f"{phase}_succeeded",
                phase=phase,
                pair_tag=pair_tag,
                attempt=attempt,
                elapsed_sec=round(result.elapsed_sec, 1),
                log_file=str(log_path),
                **merged_status,
            )
            return result

        will_retry = attempt < max_attempts
        status.emit(
            f"{phase}_failed",
            phase=phase,
            pair_tag=pair_tag,
            attempt=attempt,
            elapsed_sec=round(result.elapsed_sec, 1),
            returncode=result.returncode,
            will_retry=will_retry,
            log_file=str(log_path),
            output_tail=result.output.splitlines()[-20:],
            **merged_status,
        )
        if will_retry:
            time.sleep(retry_backoff_sec * attempt)

    raise RuntimeError(f"{phase} failed after {max_attempts} attempts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Optional source::target override. Repeat to define an explicit overnight pair list.",
    )
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--results-dir", default="results/overnight_rotalign")
    parser.add_argument("--checkpoint-dir", default="checkpoints/overnight_rotalign")
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--retry-backoff-sec", type=int, default=30)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--alignment",
        default="auto",
        choices=["auto", "identity", "procrustes", "procrustes_rand", "ridge", "cca", "reduced_rank"],
    )
    parser.add_argument(
        "--rotation",
        default="orthogonal",
        choices=["identity", "orthogonal", "hadamard"],
    )
    parser.set_defaults(whitening=True)
    parser.add_argument("--no-whitening", dest="whitening", action="store_false")
    parser.add_argument(
        "--layer-pairing",
        default="interp",
        choices=["interp", "cka"],
    )
    parser.add_argument("--selection-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--fixed-gate", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["target", "t2t", "rotalign"],
        choices=[
            "target",
            "source",
            "routing",
            "t2t",
            "rotalign",
            "rotalign_translated",
            "rotalign_fused",
            "rotalign_text_kv",
        ],
    )
    parser.add_argument("--task-type", default="generation", choices=["auto", "mcq", "generation"])
    parser.add_argument("--primary-metric", default="rotalign_kv")
    parser.set_defaults(reuse_checkpoints=True)
    parser.add_argument("--no-reuse-checkpoints", dest="reuse_checkpoints", action="store_false")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--min-ablation-seconds-left", type=int, default=3 * 60 * 60)
    parser.add_argument("--ablation-runtime-multiplier", type=float, default=4.5)
    return parser.parse_args()


def resolve_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.pair:
        return DEFAULT_PAIRS
    pairs: list[tuple[str, str]] = []
    for item in args.pair:
        if "::" not in item:
            raise ValueError(f"Invalid --pair value {item!r}; expected source::target")
        source_model, target_model = item.split("::", 1)
        pairs.append((source_model.strip(), target_model.strip()))
    return pairs


def build_calibrate_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    pair: tuple[str, str],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    source_model, target_model = pair
    cmd = [
        python_exe,
        str(repo_root / "scripts" / "calibrate.py"),
        "--source-model",
        source_model,
        "--target-model",
        target_model,
        "--calibration-file",
        args.calibration_file,
        "--output",
        str(checkpoint_path),
        "--bits",
        str(args.bits),
        "--rotation",
        args.rotation,
        "--alignment",
        args.alignment,
        "--layer-pairing",
        args.layer_pairing,
        "--layer-selection-ratio",
        str(args.selection_ratio),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    if args.whitening:
        cmd.append("--whitening")
    return cmd


def build_evaluate_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    pair: tuple[str, str],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    source_model, target_model = pair
    return [
        python_exe,
        str(repo_root / "scripts" / "evaluate.py"),
        "--translator",
        str(checkpoint_path),
        "--source-model",
        source_model,
        "--target-model",
        target_model,
        "--eval-file",
        args.eval_file,
        "--task-type",
        args.task_type,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--methods",
        *args.methods,
        "--gate-mode",
        "checkpoint",
    ]


def build_ablation_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    pair: tuple[str, str],
    args: argparse.Namespace,
    ablation_output: Path,
    ablation_checkpoint_dir: Path,
) -> list[str]:
    source_model, target_model = pair
    return [
        python_exe,
        str(repo_root / "scripts" / "ablation_sweep.py"),
        "--source-model",
        source_model,
        "--target-model",
        target_model,
        "--calibration-file",
        args.calibration_file,
        "--eval-file",
        args.eval_file,
        "--output",
        str(ablation_output),
        "--checkpoint-dir",
        str(ablation_checkpoint_dir),
        "--rotations",
        "identity",
        "orthogonal",
        "--alignments",
        "identity",
        "ridge",
        "--bits",
        str(args.bits),
        "--whiten",
        "on" if args.whitening else "off",
        "--layer-pairings",
        "interp",
        "--selection-ratios",
        str(args.selection_ratio),
        "--rotation-seeds",
        str(args.seed),
        "--protocols",
        "fused",
        "--gate-mode",
        "checkpoint",
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]


def choose_best_pair(records: list[dict[str, Any]], primary_metric: str) -> dict[str, Any] | None:
    scored = [record for record in records if primary_metric in record.get("metrics", {})]
    if not scored:
        return None
    return max(scored, key=lambda record: record["metrics"][primary_metric])


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pairs = resolve_pairs(args)

    calibration_file = Path(args.calibration_file)
    eval_file = Path(args.eval_file)
    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    results_dir = Path(args.results_dir)
    logs_dir = results_dir / "logs"
    checkpoints_dir = Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    status_file = Path(args.status_file) if args.status_file else results_dir / "status.jsonl"
    status = StatusWriter(status_file)

    started = time.monotonic()
    deadline = started + args.budget_hours * 60 * 60
    python_exe = sys.executable

    status.emit(
        "batch_started",
        phase="batch",
        repo_root=str(repo_root),
        calibration_file=str(calibration_file),
        eval_file=str(eval_file),
        results_dir=str(results_dir),
        checkpoint_dir=str(checkpoints_dir),
        status_file=str(status_file),
        device=args.device,
        dtype=args.dtype,
        methods=args.methods,
        budget_hours=args.budget_hours,
        pairs=[{"source_model": src, "target_model": tgt} for src, tgt in pairs],
    )

    completed_pairs: list[dict[str, Any]] = []
    write_pair_summaries(completed_pairs, args.primary_metric, results_dir)
    for pair_index, pair in enumerate(pairs, start=1):
        source_model, target_model = pair
        pair_tag = f"{pair_index:02d}_{slugify(source_model)}__to__{slugify(target_model)}"
        pair_log = logs_dir / f"{pair_tag}.log"
        checkpoint_path = checkpoints_dir / f"{pair_tag}.pt"
        pair_started = time.monotonic()

        status.emit(
            "pair_started",
            phase="pair",
            pair_tag=pair_tag,
            pair_index=pair_index,
            source_model=source_model,
            target_model=target_model,
            checkpoint_path=str(checkpoint_path),
            log_file=str(pair_log),
            seconds_left=round(deadline - time.monotonic(), 1),
        )

        try:
            calibration_status = {
                "pair_index": pair_index,
                "source_model": source_model,
                "target_model": target_model,
                "checkpoint_path": str(checkpoint_path),
            }
            if args.reuse_checkpoints and checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
                status.emit(
                    "calibration_reused",
                    phase="calibration",
                    pair_tag=pair_tag,
                    log_file=str(pair_log),
                    **calibration_status,
                )
            else:
                calibrate_cmd = build_calibrate_cmd(
                    python_exe=python_exe,
                    repo_root=repo_root,
                    pair=pair,
                    checkpoint_path=checkpoint_path,
                    args=args,
                )
                run_with_retries(
                    cmd=calibrate_cmd,
                    cwd=repo_root,
                    log_path=pair_log,
                    status=status,
                    phase="calibration",
                    pair_tag=pair_tag,
                    max_attempts=args.retry_count,
                    retry_backoff_sec=args.retry_backoff_sec,
                    extra_status=calibration_status,
                )

            evaluate_cmd = build_evaluate_cmd(
                python_exe=python_exe,
                repo_root=repo_root,
                pair=pair,
                checkpoint_path=checkpoint_path,
                args=args,
            )
            eval_result = run_with_retries(
                cmd=evaluate_cmd,
                cwd=repo_root,
                log_path=pair_log,
                status=status,
                phase="evaluation",
                pair_tag=pair_tag,
                max_attempts=args.retry_count,
                retry_backoff_sec=args.retry_backoff_sec,
                extra_status={
                    "pair_index": pair_index,
                    "source_model": source_model,
                    "target_model": target_model,
                    "checkpoint_path": str(checkpoint_path),
                },
            )

            metrics = parse_summary_metrics(eval_result.output)
            pair_elapsed = time.monotonic() - pair_started
            pair_record = {
                "pair_index": pair_index,
                "pair_tag": pair_tag,
                "source_model": source_model,
                "target_model": target_model,
                "checkpoint_path": str(checkpoint_path),
                "log_file": str(pair_log),
                "elapsed_sec": round(pair_elapsed, 1),
                "metrics": metrics,
            }
            completed_pairs.append(pair_record)
            write_pair_summaries(completed_pairs, args.primary_metric, results_dir)
            status.emit(
                "pair_completed",
                phase="pair",
                **pair_record,
            )
        except Exception as exc:
            status.emit(
                "pair_failed",
                phase="pair",
                pair_tag=pair_tag,
                pair_index=pair_index,
                source_model=source_model,
                target_model=target_model,
                checkpoint_path=str(checkpoint_path),
                log_file=str(pair_log),
                elapsed_sec=round(time.monotonic() - pair_started, 1),
                error=str(exc),
            )

    seconds_left = deadline - time.monotonic()
    avg_pair_sec = (
        sum(record["elapsed_sec"] for record in completed_pairs) / len(completed_pairs)
        if completed_pairs
        else 0.0
    )
    estimated_ablation_sec = max(
        float(args.min_ablation_seconds_left),
        avg_pair_sec * args.ablation_runtime_multiplier,
    )

    if args.skip_ablation:
        status.emit(
            "ablation_skipped",
            phase="ablation",
            reason="skip_ablation_flag",
            seconds_left=round(seconds_left, 1),
        )
    elif not completed_pairs:
        status.emit(
            "ablation_skipped",
            phase="ablation",
            reason="no_completed_pairs",
            seconds_left=round(seconds_left, 1),
        )
    elif seconds_left < estimated_ablation_sec:
        status.emit(
            "ablation_skipped",
            phase="ablation",
            reason="insufficient_time_budget",
            seconds_left=round(seconds_left, 1),
            estimated_ablation_sec=round(estimated_ablation_sec, 1),
        )
    else:
        best_pair = choose_best_pair(completed_pairs, args.primary_metric)
        if best_pair is None:
            status.emit(
                "ablation_skipped",
                phase="ablation",
                reason="primary_metric_missing",
                primary_metric=args.primary_metric,
                seconds_left=round(seconds_left, 1),
            )
        else:
            ablation_tag = f"{best_pair['pair_tag']}__tiny_ablation"
            ablation_log = logs_dir / f"{ablation_tag}.log"
            ablation_output = results_dir / f"{ablation_tag}.jsonl"
            ablation_checkpoint_dir = checkpoints_dir / ablation_tag
            pair = (best_pair["source_model"], best_pair["target_model"])
            ablation_cmd = build_ablation_cmd(
                python_exe=python_exe,
                repo_root=repo_root,
                pair=pair,
                args=args,
                ablation_output=ablation_output,
                ablation_checkpoint_dir=ablation_checkpoint_dir,
            )

            status.emit(
                "ablation_started",
                phase="ablation",
                pair_tag=best_pair["pair_tag"],
                source_model=best_pair["source_model"],
                target_model=best_pair["target_model"],
                output_path=str(ablation_output),
                log_file=str(ablation_log),
                seconds_left=round(seconds_left, 1),
                primary_metric=args.primary_metric,
            )
            try:
                if ablation_output.exists():
                    ablation_output.unlink()
                run_with_retries(
                    cmd=ablation_cmd,
                    cwd=repo_root,
                    log_path=ablation_log,
                    status=status,
                    phase="ablation_run",
                    pair_tag=best_pair["pair_tag"],
                    max_attempts=args.retry_count,
                    retry_backoff_sec=args.retry_backoff_sec,
                    extra_status={
                        "source_model": best_pair["source_model"],
                        "target_model": best_pair["target_model"],
                        "output_path": str(ablation_output),
                        "log_file": str(ablation_log),
                    },
                )
                ablation_records = read_jsonl(ablation_output)
                best_ablation = None
                if ablation_records:
                    best_ablation = max(
                        (
                            record
                            for record in ablation_records
                            if isinstance(record.get(args.primary_metric), (int, float))
                        ),
                        key=lambda record: float(record[args.primary_metric]),
                        default=None,
                    )
                status.emit(
                    "ablation_completed",
                    phase="ablation",
                    pair_tag=best_pair["pair_tag"],
                    source_model=best_pair["source_model"],
                    target_model=best_pair["target_model"],
                    output_path=str(ablation_output),
                    log_file=str(ablation_log),
                    num_configs=len(ablation_records),
                    best_record=best_ablation,
                )
            except Exception as exc:
                status.emit(
                    "ablation_failed",
                    phase="ablation",
                    pair_tag=best_pair["pair_tag"],
                    source_model=best_pair["source_model"],
                    target_model=best_pair["target_model"],
                    output_path=str(ablation_output),
                    log_file=str(ablation_log),
                    error=str(exc),
                )

    status.emit(
        "batch_completed",
        phase="batch",
        elapsed_sec=round(time.monotonic() - started, 1),
        completed_pairs=len(completed_pairs),
        status_file=str(status_file),
    )


if __name__ == "__main__":
    main()
