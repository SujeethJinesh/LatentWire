#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRNAME = "_artifacts"


@dataclass(frozen=True)
class ContractPaths:
    results_dir: pathlib.Path
    slice_size: int
    eval_stem: str = "gsm8k_eval"
    artifact_dir_name: str = DEFAULT_ARTIFACT_DIRNAME

    @property
    def artifacts_dir(self) -> pathlib.Path:
        return self.results_dir / self.artifact_dir_name

    @property
    def materialized_eval_file(self) -> pathlib.Path:
        return self.artifacts_dir / f"{self.eval_stem}_{self.slice_size}.jsonl"


def resolve_materialized_eval_file(
    requested: str | pathlib.Path | None,
    *,
    results_dir: pathlib.Path,
    slice_size: int,
    eval_stem: str = "gsm8k_eval",
) -> pathlib.Path:
    if requested:
        return pathlib.Path(requested)
    return ContractPaths(
        results_dir=results_dir,
        slice_size=slice_size,
        eval_stem=eval_stem,
    ).materialized_eval_file


def python_executable(root: pathlib.Path = ROOT) -> str:
    for candidate in (
        root / "venv_arm64" / "bin" / "python",
        root / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def default_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cache_root = ROOT / ".hf_home"
    env.setdefault("HF_HOME", str(cache_root))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    env.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hub"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run(
    cmd: list[str],
    *,
    cwd: pathlib.Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    env = default_env()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def chat_template_cli_args(*, enabled: bool, thinking: bool) -> list[str]:
    if not enabled:
        return []
    thinking_value = "true" if thinking else "false"
    return [
        "--source-use-chat-template",
        "--target-use-chat-template",
        "--source-enable-thinking",
        thinking_value,
        "--target-enable-thinking",
        thinking_value,
    ]


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def materialize_slice(src: pathlib.Path, dst: pathlib.Path, limit: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as in_handle, dst.open("w") as out_handle:
        for idx, line in enumerate(in_handle):
            if idx >= limit:
                break
            out_handle.write(line)


def load_sidecar(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())


def normalize_method_name(name: str) -> str:
    if name in {"target_alone", "text_to_text"}:
        return name
    if name.startswith("rotalign_kv"):
        return "rotalign_kv"
    if name.startswith("c2c"):
        return "c2c_generate"
    return name


def group_by_method(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(normalize_method_name(str(record["method"])), []).append(record)
    return grouped


_NUMBER_RE = re.compile(r"-?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?")
_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_EXPLICIT_NUMERIC_PATTERNS = (
    re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"answer\s*(?:is|=|:)\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
)


def stable_example_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


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
                    "example_id": stable_example_id(
                        {
                            "prompt": str(prompt),
                            "answers": [str(raw_answer), *[str(alias) for alias in aliases]],
                        }
                    ),
                }
            )
    return items


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


def accuracy_from_records(records: list[dict[str, Any]], eval_examples_path: pathlib.Path) -> float:
    examples = load_generation(str(eval_examples_path))
    correct = 0
    for record, example in zip(records, examples):
        correct += int(_generation_match(str(record["prediction"]), example["answers"]))
    return float(correct / max(len(records), 1))


def attach_prompts(records: list[dict[str, Any]], eval_examples_path: pathlib.Path) -> list[dict[str, Any]]:
    examples = load_generation(str(eval_examples_path))
    by_method = group_by_method(records)
    annotated: list[dict[str, Any]] = []
    for _, method_records in by_method.items():
        if len(method_records) != len(examples):
            raise ValueError("Method record count does not match eval slice size")
        for record, example in zip(method_records, examples):
            annotated.append({**record, "prompt": example["prompt"]})
    return annotated


def method_row(records: list[dict[str, Any]]) -> dict[str, Any]:
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


def paired_vs_baseline(
    method_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
) -> dict[str, int]:
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
