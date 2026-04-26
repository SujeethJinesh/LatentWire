#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_common as harness


DEFAULT_EVAL_FILE = pathlib.Path("data/svamp_eval_70.jsonl")
DEFAULT_TRANSLATOR = pathlib.Path("checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt")
DEFAULT_SOURCE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-0.6B"
METHOD_OUTPUTS = {
    "source": "source_alone.jsonl",
    "target": "target_alone.jsonl",
    "t2t": "text_to_text.jsonl",
    "c2c": "c2c_generate.jsonl",
}
METHOD_RECORD_NAMES = {
    "source": "source_alone",
    "target": "target_alone",
    "t2t": "text_to_text",
    "c2c": "c2c_generate",
}


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _count_jsonl(path: pathlib.Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_generation_eval_command(
    *,
    method: str,
    eval_file: pathlib.Path,
    prediction_output: pathlib.Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        harness.python_executable(ROOT),
        str(ROOT / "latent_bridge" / "evaluate.py"),
        "--translator",
        str(_resolve(args.translator)),
        "--source-model",
        args.source_model,
        "--target-model",
        args.target_model,
        "--eval-file",
        str(eval_file),
        "--task-type",
        "generation",
        "--device",
        args.device,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--source-reasoning-mode",
        args.source_reasoning_mode,
        "--methods",
        method,
        "--prediction-output",
        str(prediction_output),
    ]
    command.extend(
        harness.chat_template_cli_args(
            enabled=bool(args.use_chat_template),
            thinking=bool(args.enable_thinking),
        )
    )
    return command


def build_c2c_command(
    *,
    eval_file: pathlib.Path,
    prediction_output: pathlib.Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        harness.python_executable(ROOT),
        str(ROOT / "scripts" / "run_c2c_eval.py"),
        "--source-model",
        args.source_model,
        "--target-model",
        args.target_model,
        "--eval-file",
        str(eval_file),
        "--device",
        args.device,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prediction-output",
        str(prediction_output),
    ]


def _command_for_method(
    *,
    method: str,
    eval_file: pathlib.Path,
    prediction_output: pathlib.Path,
    args: argparse.Namespace,
) -> list[str]:
    if method == "c2c":
        return build_c2c_command(
            eval_file=eval_file,
            prediction_output=prediction_output,
            args=args,
        )
    return build_generation_eval_command(
        method=method,
        eval_file=eval_file,
        prediction_output=prediction_output,
        args=args,
    )


def _run_logged(command: list[str], *, log_path: pathlib.Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            env=harness.default_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - start
    return {
        "returncode": int(result.returncode),
        "elapsed_sec": float(elapsed),
    }


def _records_for_expected_method(path: pathlib.Path, expected_method: str) -> list[dict[str, Any]]:
    grouped = harness.group_by_method(harness.read_jsonl(path))
    if expected_method in grouped:
        return grouped[expected_method]
    raise KeyError(f"Expected method {expected_method!r} in {path}; found {sorted(grouped)}")


def _summarize_record_file(
    *,
    method: str,
    path: pathlib.Path,
    expected_ids: list[str],
) -> dict[str, Any]:
    expected_method = METHOD_RECORD_NAMES[method]
    records = _records_for_expected_method(path, expected_method)
    ordered_ids = [str(record.get("example_id")) for record in records]
    correct_ids = {str(record.get("example_id")) for record in records if bool(record.get("correct"))}
    total = len(records)
    correct = len(correct_ids)
    return {
        "method": method,
        "record_method": expected_method,
        "path": _display_path(path),
        "n": total,
        "correct": correct,
        "accuracy": float(correct / max(total, 1)),
        "empty_predictions": int(sum(int(not str(row.get("prediction", "")).strip()) for row in records)),
        "numeric_extraction_coverage": int(
            sum(int(harness._has_numeric_extraction(str(row.get("prediction", "")))) for row in records)
        ),
        "ordered_example_ids": ordered_ids,
        "exact_id_parity": ordered_ids == expected_ids,
        "set_id_parity": set(ordered_ids) == set(expected_ids),
        "unique_example_ids": len(set(ordered_ids)) == len(ordered_ids),
        "correct_ids": sorted(correct_ids),
    }


def _same_resolved_path(left: Any, right: pathlib.Path) -> bool:
    if left is None:
        return False
    try:
        return pathlib.Path(str(left)).resolve() == right.resolve()
    except OSError:
        return False


def _expected_sidecar_config(
    *,
    method: str,
    materialized_eval_file: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "source_model": args.source_model,
        "target_model": args.target_model,
        "device": args.device,
        "max_new_tokens": int(args.max_new_tokens),
        "eval_file": materialized_eval_file,
    }
    if method == "c2c":
        config["baseline"] = "c2c"
    else:
        thinking_value = "true" if bool(args.enable_thinking) else "false"
        if not bool(args.use_chat_template):
            thinking_value = "auto"
        config.update(
            {
                "translator": _resolve(args.translator),
                "source_reasoning_mode": args.source_reasoning_mode,
                "source_use_chat_template": bool(args.use_chat_template),
                "target_use_chat_template": bool(args.use_chat_template),
                "source_enable_thinking": thinking_value,
                "target_enable_thinking": thinking_value,
                "methods": [method],
            }
        )
    return config


def _validate_sidecar_config(
    *,
    method: str,
    path: pathlib.Path,
    materialized_eval_file: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    sidecar_path = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar_path.exists():
        return False, "missing_sidecar"
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - validation should explain corrupt artifacts.
        return False, f"invalid_sidecar:{exc}"
    run_config = sidecar.get("run_config")
    if not isinstance(run_config, dict):
        return False, "missing_run_config"
    expected = _expected_sidecar_config(
        method=method,
        materialized_eval_file=materialized_eval_file,
        args=args,
    )
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual = run_config.get(key)
        if isinstance(expected_value, pathlib.Path):
            if not _same_resolved_path(actual, expected_value):
                mismatches.append(key)
        elif actual != expected_value:
            mismatches.append(key)
    if mismatches:
        return False, "config_mismatch:" + ",".join(mismatches)
    return True, "ok"


def _validate_record_file(
    *,
    method: str,
    path: pathlib.Path,
    expected_ids: list[str],
    materialized_eval_file: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, Any] | None]:
    if not path.exists():
        return False, "missing", None
    try:
        summary = _summarize_record_file(
            method=method,
            path=path,
            expected_ids=expected_ids,
        )
    except Exception as exc:  # noqa: BLE001 - validation should explain corrupt artifacts.
        return False, f"invalid: {exc}", None
    if int(summary["n"]) != len(expected_ids):
        return False, f"wrong_count:{summary['n']}!=expected:{len(expected_ids)}", summary
    if not bool(summary["exact_id_parity"]):
        return False, "ordered_id_mismatch", summary
    if not bool(summary["unique_example_ids"]):
        return False, "duplicate_example_ids", summary
    sidecar_valid, sidecar_status = _validate_sidecar_config(
        method=method,
        path=path,
        materialized_eval_file=materialized_eval_file,
        args=args,
    )
    summary["sidecar_config_validation"] = sidecar_status
    if not sidecar_valid:
        return False, sidecar_status, summary
    return True, "ok", summary


def _temp_prediction_path(path: pathlib.Path, *, method: str) -> pathlib.Path:
    return path.with_name(f".{path.name}.tmp.{method}.{os.getpid()}")


def _replace_prediction_artifact(temp_output: pathlib.Path, final_output: pathlib.Path) -> None:
    temp_sidecar = temp_output.with_suffix(temp_output.suffix + ".meta.json")
    final_sidecar = final_output.with_suffix(final_output.suffix + ".meta.json")
    temp_output.replace(final_output)
    if temp_sidecar.exists():
        temp_sidecar.replace(final_sidecar)


def _pairwise_vs_target(method_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if "target" not in method_summaries:
        return []
    target = method_summaries["target"]
    target_correct = set(target["correct_ids"])
    pairs: list[dict[str, Any]] = []
    for method, summary in sorted(method_summaries.items()):
        if method == "target":
            continue
        method_correct = set(summary["correct_ids"])
        oracle = target_correct | method_correct
        pairs.append(
            {
                "method": method,
                "target_method": "target",
                "n": int(target["n"]),
                "method_only_count": len(method_correct - target_correct),
                "target_only_count": len(target_correct - method_correct),
                "both_correct_count": len(method_correct & target_correct),
                "oracle_count": len(oracle),
                "oracle_accuracy": float(len(oracle) / max(int(target["n"]), 1)),
                "method_only_ids": sorted(method_correct - target_correct),
                "target_only_ids": sorted(target_correct - method_correct),
            }
        )
    return pairs


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Generation Baseline Materialization",
        "",
        f"- date: `{payload['date']}`",
        f"- eval file: `{payload['eval_file']}`",
        f"- materialized eval file: `{payload['materialized_eval_file']}`",
        f"- limit: `{payload['limit']}`",
        f"- dry run: `{payload['dry_run']}`",
        "",
        "## Run Rows",
        "",
        "| Method | Status | Return | Output | Log |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| `{row['method']}` | `{row['status']}` | "
            f"{row.get('returncode', '')} | `{row['prediction_output']}` | `{row['log']}` |"
        )
    lines.extend(
        [
            "",
            "## Method Summaries",
            "",
            "| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["method_summaries"].values():
        lines.append(
            f"| `{row['method']}` | {row['correct']}/{row['n']} | "
            f"{row['accuracy']:.3f} | `{row['exact_id_parity']}` | "
            f"{row['numeric_extraction_coverage']}/{row['n']} | {row['empty_predictions']} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Versus Target",
            "",
            "| Method | Method-only | Target-only | Both correct | Oracle |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["pairwise_vs_target"]:
        lines.append(
            f"| `{row['method']}` | {row['method_only_count']} | "
            f"{row['target_only_count']} | {row['both_correct_count']} | "
            f"{row['oracle_count']}/{row['n']} |"
        )
    lines.extend(["", "## Commands", ""])
    for row in payload["runs"]:
        lines.append(f"### {row['method']}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(row["command"]))
        lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    eval_file = _resolve(args.eval_file)
    results_dir = _resolve(args.results_dir)
    output_json = _resolve(args.output_json) if args.output_json else results_dir / "manifest.json"
    output_md = _resolve(args.output_md) if args.output_md else results_dir / "manifest.md"
    artifact_dir = results_dir / "_artifacts"
    log_dir = results_dir / "logs"
    materialized_eval_file = artifact_dir / f"{eval_file.stem}_{args.limit}.jsonl"

    harness.materialize_slice(eval_file, materialized_eval_file, int(args.limit))
    actual_count = _count_jsonl(materialized_eval_file)
    if actual_count != int(args.limit):
        raise ValueError(
            f"Materialized {actual_count} rows from {eval_file}, expected --limit={args.limit}"
        )
    expected_ids = [
        str(example["example_id"])
        for example in harness.load_generation(str(materialized_eval_file))
    ]

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for method in args.methods:
        prediction_output = results_dir / METHOD_OUTPUTS[method]
        log_path = log_dir / f"{method}.log"
        existing_valid, validation, _ = _validate_record_file(
            method=method,
            path=prediction_output,
            expected_ids=expected_ids,
            materialized_eval_file=materialized_eval_file,
            args=args,
        )
        command_output = (
            prediction_output
            if existing_valid or args.dry_run
            else _temp_prediction_path(prediction_output, method=method)
        )
        command = _command_for_method(
            method=method,
            eval_file=materialized_eval_file,
            prediction_output=command_output,
            args=args,
        )
        row: dict[str, Any] = {
            "method": method,
            "prediction_output": _display_path(prediction_output),
            "command_prediction_output": _display_path(command_output),
            "log": _display_path(log_path),
            "command": command,
            "existing_output_validation": validation,
        }
        if existing_valid and not args.force:
            row["status"] = "skipped_existing"
        elif args.dry_run:
            row["status"] = "dry_run"
        else:
            result = _run_logged(command, log_path=log_path)
            row.update(result)
            if result["returncode"] == 0:
                temp_valid, temp_validation, _ = _validate_record_file(
                    method=method,
                    path=command_output,
                    expected_ids=expected_ids,
                    materialized_eval_file=materialized_eval_file,
                    args=args,
                )
                row["temp_output_validation"] = temp_validation
                if temp_valid:
                    _replace_prediction_artifact(command_output, prediction_output)
                    row["status"] = "ran"
                else:
                    row["status"] = "failed_validation"
                    failures.append(row)
            else:
                row["status"] = "failed"
                failures.append(row)
        runs.append(row)

    method_summaries: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        prediction_output = results_dir / METHOD_OUTPUTS[method]
        _, _, summary = _validate_record_file(
            method=method,
            path=prediction_output,
            expected_ids=expected_ids,
            materialized_eval_file=materialized_eval_file,
            args=args,
        )
        if summary is not None:
            method_summaries[method] = summary

    payload = {
        "date": str(date.today()),
        "claim_policy": (
            "dev-smoke when --limit is below the frozen decision size; "
            "paper-eligible only after rerunning the frozen gate with seed/control protocol"
        ),
        "dry_run": bool(args.dry_run),
        "eval_file": _display_path(eval_file),
        "eval_file_sha256": _sha256_file(eval_file),
        "materialized_eval_file": _display_path(materialized_eval_file),
        "materialized_eval_file_sha256": _sha256_file(materialized_eval_file),
        "results_dir": _display_path(results_dir),
        "translator": _display_path(_resolve(args.translator)),
        "translator_sha256": _sha256_file(_resolve(args.translator)) if _resolve(args.translator).exists() else None,
        "limit": int(args.limit),
        "source_model": args.source_model,
        "target_model": args.target_model,
        "device": args.device,
        "max_new_tokens": int(args.max_new_tokens),
        "methods": list(args.methods),
        "runs": runs,
        "method_summaries": method_summaries,
        "pairwise_vs_target": _pairwise_vs_target(method_summaries),
    }
    _write_json(output_json, payload)
    _write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures and not args.continue_on_error:
        failed_methods = ", ".join(str(row["method"]) for row in failures)
        raise RuntimeError(f"Failed materialization methods: {failed_methods}")
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize exact-ID generation baselines one method at a time."
    )
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--results-dir", default=f"results/svamp_exactid_baselines_{date.today():%Y%m%d}")
    parser.add_argument("--translator", default=str(DEFAULT_TRANSLATOR))
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_OUTPUTS), default=["source", "target", "t2t", "c2c"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--source-reasoning-mode",
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        default="brief_analysis",
    )
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run(_parse_args(argv))


if __name__ == "__main__":
    main()
