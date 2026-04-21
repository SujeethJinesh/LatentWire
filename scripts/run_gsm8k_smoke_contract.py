#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import hashlib
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SOURCE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_CHECKPOINT = "checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt"
DEFAULT_EVAL_FILE = "data/gsm8k_eval_70.jsonl"


@dataclass(frozen=True)
class GSM8KSmokeConfig:
    source_model: str = DEFAULT_SOURCE_MODEL
    target_model: str = DEFAULT_TARGET_MODEL
    checkpoint_path: str = DEFAULT_CHECKPOINT
    eval_file: str = DEFAULT_EVAL_FILE
    slice_size: int = 32
    materialized_eval_file: str = "/tmp/gsm8k_eval_32.jsonl"
    results_dir: str = "results/gsm8k_smoke_contract_20260421"
    device: str = "mps"
    max_new_tokens: int = 64
    gate: float = 0.10
    kv_transport: str = "k_only"
    position_selection_ratio: float = 0.5
    position_selection_metric: str = "attention"
    source_reasoning_mode: str = "brief_analysis"
    use_chat_template: bool = True
    enable_thinking: bool = False


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _materialize_slice(src: pathlib.Path, dst: pathlib.Path, limit: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as in_handle, dst.open("w") as out_handle:
        for idx, line in enumerate(in_handle):
            if idx >= limit:
                break
            out_handle.write(line)


def _default_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cache_root = ROOT / ".hf_home"
    env.setdefault("HF_HOME", str(cache_root))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    env.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hub"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _run(cmd: list[str], *, cwd: pathlib.Path) -> None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_default_env(),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _load_sidecar(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())


def _normalize_method_name(name: str) -> str:
    if name == "target_alone":
        return name
    if name == "text_to_text":
        return name
    if name.startswith("rotalign_kv"):
        return "rotalign_kv"
    if name.startswith("c2c"):
        return "c2c_generate"
    return name


def _group_by_method(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_normalize_method_name(str(record["method"])), []).append(record)
    return grouped


_NUMBER_RE = re.compile(r"-?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?")
_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_EXPLICIT_NUMERIC_PATTERNS = (
    re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"answer\s*(?:is|=|:)\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
)


def load_generation(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("question")
            raw_answer = obj.get("answer_text", obj.get("answer", obj.get("target")))
            aliases = obj.get("aliases", [])
            items.append(
                {
                    "prompt": str(prompt),
                    "answers": [str(raw_answer), *[str(alias) for alias in aliases]],
                    "example_id": _stable_example_id(
                        {
                            "prompt": str(prompt),
                            "answers": [str(raw_answer), *[str(alias) for alias in aliases]],
                        }
                    ),
                }
            )
    return items


def _stable_example_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _normalize_generation_text(text: str) -> str:
    norm = " ".join(text.strip().lower().split())
    norm = re.sub(r"^[`'\"“”‘’\(\[]+", "", norm)
    norm = re.sub(r"[`'\"“”‘’\)\]\.!,?:;]+$", "", norm)
    return norm


def _normalize_numeric_string(text: str) -> str | None:
    cleaned = text.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(".,!?;: ")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized


def _extract_reference_numeric_answer(answer: str) -> str | None:
    for pattern in _EXPLICIT_NUMERIC_PATTERNS:
        matches = pattern.findall(answer)
        if matches:
            numeric = _normalize_numeric_string(str(matches[-1]))
            if numeric is not None:
                return numeric
    return _normalize_numeric_string(answer)


def _extract_prediction_numeric_answer(prediction: str) -> str | None:
    for pattern in _EXPLICIT_NUMERIC_PATTERNS:
        matches = pattern.findall(prediction)
        if matches:
            numeric = _normalize_numeric_string(str(matches[-1]))
            if numeric is not None:
                return numeric
    candidates = _NUMERIC_TOKEN_RE.findall(prediction)
    for candidate in reversed(candidates):
        numeric = _normalize_numeric_string(candidate)
        if numeric is not None:
            return numeric
    return None


def _generation_match(prediction: str, answers: list[str]) -> bool:
    norm_pred = _normalize_generation_text(prediction)
    normalized_answers = {_normalize_generation_text(answer) for answer in answers}
    if norm_pred in normalized_answers:
        return True
    numeric_answers = {
        numeric
        for answer in answers
        if (numeric := _extract_reference_numeric_answer(answer)) is not None
    }
    if not numeric_answers:
        return False
    pred_numeric = _extract_prediction_numeric_answer(prediction)
    return pred_numeric in numeric_answers


def _has_numeric_extraction(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))


def _accuracy_from_records(records: list[dict[str, Any]], eval_examples_path: pathlib.Path) -> float:
    examples = load_generation(str(eval_examples_path))
    correct = 0
    for record, example in zip(records, examples):
        correct += int(_generation_match(str(record["prediction"]), example["answers"]))
    return float(correct / max(len(records), 1))


def _attach_prompts(records: list[dict[str, Any]], eval_examples_path: pathlib.Path) -> list[dict[str, Any]]:
    examples = load_generation(str(eval_examples_path))
    by_method = _group_by_method(records)
    annotated: list[dict[str, Any]] = []
    for _, method_records in by_method.items():
        if len(method_records) != len(examples):
            raise ValueError("Method record count does not match eval slice size")
        for record, example in zip(method_records, examples):
            annotated.append({**record, "prompt": example["prompt"]})
    return annotated


def _method_row(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(int(record.get("correct", False)) for record in records)
    generated_tokens = [int(record.get("generated_tokens", 0)) for record in records]
    latency = [float(record.get("latency_sec", 0.0)) for record in records]
    example_ids = [str(record.get("example_id")) for record in records]
    numeric_coverage = sum(int(_has_numeric_extraction(str(record.get("prediction", "")))) for record in records)
    return {
        "n": total,
        "accuracy": float(correct / max(total, 1)),
        "generated_tokens_avg": float(sum(generated_tokens) / max(total, 1)),
        "latency_sec_avg": float(sum(latency) / max(total, 1)),
        "examples_per_sec": float(total / max(sum(latency), 1e-8)),
        "numeric_extraction_coverage": int(numeric_coverage),
        "empty_predictions": int(sum(int(not str(record.get("prediction", "")).strip()) for record in records)),
        "example_ids": example_ids,
    }


def _paired_vs_baseline(method_records: list[dict[str, Any]], baseline_records: list[dict[str, Any]]) -> dict[str, int]:
    baseline_by_id = {str(row["example_id"]): bool(row["correct"]) for row in baseline_records}
    win = 0
    loss = 0
    tie = 0
    for row in method_records:
        example_id = str(row["example_id"])
        method_correct = bool(row["correct"])
        base_correct = baseline_by_id[example_id]
        if method_correct and not base_correct:
            win += 1
        elif base_correct and not method_correct:
            loss += 1
        else:
            tie += 1
    return {"win": win, "loss": loss, "tie": tie}


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    checks = payload["checks"]
    config = payload["config"]
    lines = [
        "# GSM8K 32-Example Smoke Contract",
        "",
        f"- date: `{payload['date']}`",
        f"- source -> target: `{config['source_model']} -> {config['target_model']}`",
        f"- slice: `{config['slice_size']}` examples from `{config['eval_file']}`",
        f"- checkpoint: `{config['checkpoint_path']}`",
        "",
        "| Row | Accuracy | N | Win vs target | Loss vs target | Tie vs target | Tokens avg | Latency avg | Ex/s | Numeric coverage | Empty preds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("target_alone", "text_to_text", "rotalign_kv", "c2c_generate"):
        row = rows[name]
        paired = row.get("paired_vs_target", {"win": 0, "loss": 0, "tie": row["n"]})
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {row['n']} | {paired['win']} | {paired['loss']} | {paired['tie']} | {row['generated_tokens_avg']:.2f} | {row['latency_sec_avg']:.4f} | {row['examples_per_sec']:.4f} | {row['numeric_extraction_coverage']} | {row['empty_predictions']} |"
        )
    lines.extend(["", "## Checks", ""])
    for name, info in checks.items():
        mark = "PASS" if info["passed"] else "FAIL"
        lines.append(f"- {mark}: `{name}` — {info['detail']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_smoke(config: GSM8KSmokeConfig) -> dict[str, Any]:
    results_dir = ROOT / config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    materialized = pathlib.Path(config.materialized_eval_file)
    _materialize_slice(ROOT / config.eval_file, materialized, config.slice_size)

    py = str(ROOT / ".venv" / "bin" / "python")
    base_eval_cmd = [
        py,
        str(ROOT / "latent_bridge" / "evaluate.py"),
        "--translator",
        str(ROOT / config.checkpoint_path),
        "--source-model",
        config.source_model,
        "--target-model",
        config.target_model,
        "--eval-file",
        str(materialized),
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
    ]
    if config.use_chat_template:
        base_eval_cmd.extend(
            [
                "--source-use-chat-template",
                "--target-use-chat-template",
                "--source-enable-thinking",
                "false" if not config.enable_thinking else "true",
                "--target-enable-thinking",
                "false" if not config.enable_thinking else "true",
            ]
        )

    latentwire_output = results_dir / "gsm8k32_latentwire.jsonl"
    target_rerun_output = results_dir / "gsm8k32_target_rerun.jsonl"
    c2c_output = results_dir / "gsm8k32_c2c.jsonl"

    if not latentwire_output.exists():
        _run(
            base_eval_cmd
            + [
                "--methods",
                "target",
                "t2t",
                "rotalign",
                "--prediction-output",
                str(latentwire_output),
            ],
            cwd=ROOT,
        )
    if not target_rerun_output.exists():
        _run(
            base_eval_cmd
            + [
                "--methods",
                "target",
                "--prediction-output",
                str(target_rerun_output),
            ],
            cwd=ROOT,
        )
    if not c2c_output.exists():
        _run(
            [
                py,
                str(ROOT / "scripts" / "run_c2c_eval.py"),
                "--source-model",
                config.source_model,
                "--target-model",
                config.target_model,
                "--eval-file",
                str(materialized),
                "--device",
                config.device,
                "--max-new-tokens",
                str(config.max_new_tokens),
                "--limit",
                str(config.slice_size),
                "--prediction-output",
                str(c2c_output),
            ],
            cwd=ROOT,
        )

    latent_records = _attach_prompts(_read_jsonl(latentwire_output), materialized)
    target_rerun_records = _attach_prompts(_read_jsonl(target_rerun_output), materialized)
    c2c_records = _attach_prompts(_read_jsonl(c2c_output), materialized)

    latent_groups = _group_by_method(latent_records)
    c2c_groups = _group_by_method(c2c_records)
    rerun_groups = _group_by_method(target_rerun_records)

    rows = {
        "target_alone": _method_row(latent_groups["target_alone"]),
        "text_to_text": _method_row(latent_groups["text_to_text"]),
        "rotalign_kv": _method_row(latent_groups["rotalign_kv"]),
        "c2c_generate": _method_row(c2c_groups["c2c_generate"]),
    }
    rows["text_to_text"]["paired_vs_target"] = _paired_vs_baseline(latent_groups["text_to_text"], latent_groups["target_alone"])
    rows["rotalign_kv"]["paired_vs_target"] = _paired_vs_baseline(latent_groups["rotalign_kv"], latent_groups["target_alone"])
    rows["c2c_generate"]["paired_vs_target"] = _paired_vs_baseline(c2c_groups["c2c_generate"], latent_groups["target_alone"])
    rows["target_alone"]["paired_vs_target"] = {"win": 0, "loss": 0, "tie": rows["target_alone"]["n"]}

    latent_meta = _load_sidecar(latentwire_output)
    c2c_meta = _load_sidecar(c2c_output)
    target_rerun_meta = _load_sidecar(target_rerun_output)

    target_rerun_predictions = [str(row["prediction"]) for row in rerun_groups["target_alone"]]
    target_predictions = [str(row["prediction"]) for row in latent_groups["target_alone"]]
    target_rerun_identical = target_predictions == target_rerun_predictions

    offline_target_accuracy = _accuracy_from_records(latent_groups["target_alone"], materialized)
    target_metric_summary = float(latent_meta["metric_summary"]["target_alone"])
    c2c_accuracy = float(rows["c2c_generate"]["accuracy"])
    target_accuracy = float(rows["target_alone"]["accuracy"])
    rotalign_accuracy = float(rows["rotalign_kv"]["accuracy"])

    checks = {
        "row_counts_match_slice": {
            "passed": all(row["n"] == config.slice_size for row in rows.values()),
            "detail": f"counts={[rows[name]['n'] for name in rows]} expected={config.slice_size}",
        },
        "example_ids_identical": {
            "passed": (
                rows["target_alone"]["example_ids"] == rows["text_to_text"]["example_ids"]
                == rows["rotalign_kv"]["example_ids"]
                == rows["c2c_generate"]["example_ids"]
            ),
            "detail": "all four rows share the same ordered example IDs",
        },
        "greedy_config": {
            "passed": (
                latent_meta["run_config"]["max_new_tokens"] == config.max_new_tokens
                and c2c_meta["run_config"]["max_new_tokens"] == config.max_new_tokens
            ),
            "detail": f"max_new_tokens={config.max_new_tokens}; evaluate and c2c both run greedy generation",
        },
        "no_empty_predictions": {
            "passed": all(row["empty_predictions"] == 0 for row in rows.values()),
            "detail": f"empty_predictions={[rows[name]['empty_predictions'] for name in rows]}",
        },
        "numeric_extraction_coverage": {
            "passed": all(row["numeric_extraction_coverage"] >= config.slice_size - 1 for row in rows.values()),
            "detail": f"coverage={[rows[name]['numeric_extraction_coverage'] for name in rows]} threshold={config.slice_size - 1}",
        },
        "target_rerun_byte_identical": {
            "passed": target_rerun_identical,
            "detail": "rerun target predictions exactly match the main target row",
        },
        "target_offline_rescore_matches": {
            "passed": abs(offline_target_accuracy - target_metric_summary) < 1e-9,
            "detail": f"offline={offline_target_accuracy:.4f} sidecar={target_metric_summary:.4f}",
        },
        "c2c_beats_target_by_two": {
            "passed": c2c_accuracy >= target_accuracy + (2.0 / config.slice_size),
            "detail": f"c2c={c2c_accuracy:.4f} target={target_accuracy:.4f}",
        },
        "rotalign_ties_or_beats_target": {
            "passed": rotalign_accuracy >= target_accuracy,
            "detail": f"rotalign={rotalign_accuracy:.4f} target={target_accuracy:.4f}",
        },
    }

    payload = {
        "date": "2026-04-21",
        "config": asdict(config),
        "artifacts": {
            "materialized_eval_file": str(materialized),
            "latentwire_prediction_output": str(latentwire_output),
            "latentwire_sidecar": str(latentwire_output.with_suffix(latentwire_output.suffix + ".meta.json")),
            "target_rerun_prediction_output": str(target_rerun_output),
            "c2c_prediction_output": str(c2c_output),
            "c2c_sidecar": str(c2c_output.with_suffix(c2c_output.suffix + ".meta.json")),
        },
        "rows": rows,
        "checks": checks,
        "metric_summary": {
            "latentwire": latent_meta["metric_summary"],
            "c2c": c2c_meta["metric_summary"],
            "target_rerun": target_rerun_meta["metric_summary"],
        },
    }
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen 32-example GSM8K smoke contract.")
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--slice-size", type=int, default=32)
    parser.add_argument("--materialized-eval-file", default="/tmp/gsm8k_eval_32.jsonl")
    parser.add_argument("--results-dir", default="results/gsm8k_smoke_contract_20260421")
    parser.add_argument("--device", default="mps")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = GSM8KSmokeConfig(
        source_model=args.source_model,
        target_model=args.target_model,
        checkpoint_path=args.checkpoint_path,
        eval_file=args.eval_file,
        slice_size=args.slice_size,
        materialized_eval_file=args.materialized_eval_file,
        results_dir=args.results_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        gate=args.gate,
        kv_transport=args.kv_transport,
        position_selection_ratio=args.position_selection_ratio,
        position_selection_metric=args.position_selection_metric,
        source_reasoning_mode=args.source_reasoning_mode,
        use_chat_template=not args.no_chat_template,
        enable_thinking=args.enable_thinking,
    )
    payload = run_smoke(config)
    results_dir = ROOT / config.results_dir
    _write_json(results_dir / "gsm8k_smoke_contract_20260421.json", payload)
    _write_markdown(results_dir / "gsm8k_smoke_contract_20260421.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
