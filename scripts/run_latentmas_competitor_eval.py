#!/usr/bin/env python3
"""Run LatentMAS competitor rows on LatentWire JSONL eval files.

This wrapper deliberately stays outside the vendored LatentMAS code. It adapts
LatentWire eval rows into the item schema LatentMAS expects, then writes compact
JSONL telemetry that can be joined with our own benchmark artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence


BaselineMethod: Any | None = None
TextMASMethod: Any | None = None
LatentMASMethod: Any | None = None
ModelWrapper: Any | None = None

LATENTMAS_NATIVE_TASKS = {
    "gsm8k",
    "aime2024",
    "aime2025",
    "gpqa",
    "arc_easy",
    "arc_challenge",
    "mbppplus",
    "humanevalplus",
    "medqa",
}


@dataclass(frozen=True)
class LatentMASRunConfig:
    latentmas_root: str
    model_name: str
    method: str
    task: str
    latentmas_task: str
    prompt: str
    eval_file: str
    limit: int | None
    max_new_tokens: int
    latent_steps: int
    temperature: float
    top_p: float
    generate_bs: int
    use_vllm: bool
    latent_space_realign: bool
    seed: int
    device: str
    device2: str


def _stable_hash(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _dedupe(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def _extract_gold(row: dict[str, Any]) -> str:
    for key in ("answer_text", "answer", "gold", "target", "solution"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    raise ValueError(f"row has no answer field: {sorted(row)}")


def _extract_question(row: dict[str, Any]) -> str:
    for key in ("prompt", "question", "source_question", "input"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"row has no question field: {sorted(row)}")


def _extract_id(row: dict[str, Any], offset: int) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("id") is not None:
        return str(metadata["id"])
    for key in ("id", "example_id", "uid"):
        if row.get(key) is not None:
            return str(row[key])
    return str(offset)


def load_latentwire_generation_items(path: pathlib.Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert LatentWire JSONL eval rows into LatentMAS-compatible items."""

    items: list[dict[str, Any]] = []
    with pathlib.Path(path).open("r", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            if limit is not None and len(items) >= int(limit):
                break
            if not line.strip():
                continue
            row = json.loads(line)
            gold = _extract_gold(row)
            aliases = row.get("aliases") or []
            if not isinstance(aliases, list):
                aliases = [str(aliases)]
            items.append(
                {
                    "id": _extract_id(row, offset),
                    "question": _extract_question(row),
                    "solution": gold,
                    "gold": gold,
                    "answers": _dedupe([gold, *aliases]),
                }
            )
    return items


def _ensure_latentmas_imports(latentmas_root: pathlib.Path) -> None:
    global BaselineMethod, TextMASMethod, LatentMASMethod, ModelWrapper

    if BaselineMethod is not None and TextMASMethod is not None and LatentMASMethod is not None and ModelWrapper is not None:
        return

    root = pathlib.Path(latentmas_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"LatentMAS root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if BaselineMethod is None:
        from methods.baseline import BaselineMethod as _BaselineMethod

        BaselineMethod = _BaselineMethod
    if TextMASMethod is None:
        from methods.text_mas import TextMASMethod as _TextMASMethod

        TextMASMethod = _TextMASMethod
    if LatentMASMethod is None:
        from methods.latent_mas import LatentMASMethod as _LatentMASMethod

        LatentMASMethod = _LatentMASMethod
    if ModelWrapper is None:
        from models import ModelWrapper as _ModelWrapper

        ModelWrapper = _ModelWrapper


def make_latentmas_model(args: argparse.Namespace) -> Any:
    _ensure_latentmas_imports(pathlib.Path(args.latentmas_root))
    import torch
    from utils import auto_device, set_seed

    set_seed(int(args.seed))
    device = auto_device(args.device)
    return ModelWrapper(args.model_name, device, use_vllm=bool(args.use_vllm), args=args)


def make_latentmas_method(method_name: str, model: object, args: argparse.Namespace) -> object:
    if BaselineMethod is None or TextMASMethod is None or LatentMASMethod is None:
        _ensure_latentmas_imports(pathlib.Path(getattr(args, "latentmas_root", "references/repos/LatentMAS")))

    common_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generate_bs": args.generate_bs,
        "args": args,
    }
    if method_name == "baseline":
        return BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            use_vllm=args.use_vllm,
            **common_kwargs,
        )
    if method_name == "text_mas":
        return TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
        )
    if method_name == "latent_mas":
        return LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
    raise ValueError(f"unknown LatentMAS method: {method_name}")


def _runtime_args_for_latentmas(args: argparse.Namespace) -> argparse.Namespace:
    """Build vendor-facing args while keeping wrapper-facing task labels intact."""

    runtime_args = argparse.Namespace(**vars(args))
    explicit_task = getattr(args, "latentmas_task", None)
    if explicit_task:
        runtime_args.task = str(explicit_task)
    elif runtime_args.task not in LATENTMAS_NATIVE_TASKS:
        runtime_args.task = "gsm8k"
    runtime_args.eval_task = args.task
    return runtime_args


def _compact_agent(agent: dict[str, Any]) -> dict[str, Any]:
    input_ids = agent.get("input_ids") or []
    input_tokens = agent.get("input_tokens") or []
    if not isinstance(input_ids, list):
        input_ids = []
    if not isinstance(input_tokens, list):
        input_tokens = []

    output = agent.get("output", "")
    agent_input = agent.get("input", "")
    compact = {
        "name": agent.get("name", "Agent"),
        "role": agent.get("role", ""),
        "input_hash": _stable_hash(agent_input),
        "output_hash": _stable_hash(output),
        "input_chars": len(str(agent_input)),
        "output_chars": len(str(output)),
        "input_token_count": len(input_ids) if input_ids else len(input_tokens),
        "output_char_count": len(str(output)),
    }
    if agent.get("latent_steps") is not None:
        compact["latent_steps"] = agent.get("latent_steps")
    return compact


def _normalize_numeric_text(value: Any) -> str:
    text = str(value).strip().lower()
    if "####" in text:
        text = text.split("####", 1)[1].strip()
    text = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        numeric = matches[-1]
        if numeric.endswith(".0"):
            numeric = numeric[:-2]
        return numeric
    return re.sub(r"\s+", " ", text)


def _fallback_correct(prediction: Any, answers: Sequence[str]) -> bool:
    normalized_pred = _normalize_numeric_text(prediction)
    return any(normalized_pred == _normalize_numeric_text(answer) for answer in answers)


def _record_from_result(
    *,
    item: dict[str, Any],
    result: dict[str, Any],
    args: argparse.Namespace,
    latency_sec: float,
) -> dict[str, Any]:
    agents = result.get("agents") or []
    if not isinstance(agents, list):
        agents = []
    compact_agents = [_compact_agent(agent) for agent in agents if isinstance(agent, dict)]
    input_token_count = sum(int(agent.get("input_token_count", 0)) for agent in compact_agents)
    output_char_count = sum(int(agent.get("output_char_count", 0)) for agent in compact_agents)
    prediction = result.get("prediction")
    correct = result.get("correct")
    if correct is None:
        correct = _fallback_correct(prediction, item.get("answers", []))

    return {
        "id": item["id"],
        "question": item["question"],
        "gold": item["gold"],
        "answers": item.get("answers", []),
        "prediction": prediction,
        "raw_prediction": result.get("raw_prediction", ""),
        "correct": bool(correct),
        "method": args.method,
        "task": args.task,
        "prompt": args.prompt,
        "model_name": args.model_name,
        "latent_steps": int(args.latent_steps),
        "latency_sec": float(latency_sec),
        "trace": {
            "question_hash": _stable_hash(item["question"]),
            "raw_prediction_hash": _stable_hash(result.get("raw_prediction", "")),
            "agent_count": len(compact_agents),
            "input_token_count": input_token_count,
            "output_char_count": output_char_count,
            "agents": compact_agents,
        },
    }


def _chunks(items: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    chunk_size = max(1, int(size))
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def _call_method(method: Any, args: argparse.Namespace, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.method == "latent_mas" and bool(args.use_vllm) and hasattr(method, "run_batch_vllm"):
        return list(method.run_batch_vllm(batch))
    return list(method.run_batch(batch))


def _write_jsonl(path: pathlib.Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def _summarize(args: argparse.Namespace, records: Sequence[dict[str, Any]], config: LatentMASRunConfig) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record.get("correct"))
    latency = sum(float(record.get("latency_sec", 0.0)) for record in records)
    input_tokens = sum(int(record.get("trace", {}).get("input_token_count", 0)) for record in records)
    output_chars = sum(int(record.get("trace", {}).get("output_char_count", 0)) for record in records)
    return {
        "method": args.method,
        "task": args.task,
        "prompt": args.prompt,
        "model_name": args.model_name,
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "latency_sec": latency,
        "examples_per_sec": total / latency if latency > 0 else None,
        "input_token_count": input_tokens,
        "output_char_count": output_chars,
        "latent_steps": int(args.latent_steps),
        "prediction_output": str(args.prediction_output),
        "config": asdict(config),
    }


def run_eval(
    args: argparse.Namespace,
    *,
    model_factory: Callable[[argparse.Namespace], object] | None = None,
    method_factory: Callable[[str, object, argparse.Namespace], object] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    clock = clock or time.perf_counter
    model_factory = model_factory or make_latentmas_model
    method_factory = method_factory or make_latentmas_method
    args.prediction_output = pathlib.Path(args.prediction_output)
    args.eval_file = pathlib.Path(args.eval_file)
    args.latentmas_root = pathlib.Path(args.latentmas_root)
    if not hasattr(args, "latentmas_task"):
        args.latentmas_task = None
    runtime_args = _runtime_args_for_latentmas(args)

    config = LatentMASRunConfig(
        latentmas_root=str(args.latentmas_root),
        model_name=args.model_name,
        method=args.method,
        task=args.task,
        latentmas_task=runtime_args.task,
        prompt=args.prompt,
        eval_file=str(args.eval_file),
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        latent_steps=args.latent_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        use_vllm=args.use_vllm,
        latent_space_realign=bool(getattr(args, "latent_space_realign", False)),
        seed=args.seed,
        device=args.device,
        device2=args.device2,
    )

    items = load_latentwire_generation_items(args.eval_file, limit=args.limit)
    model = model_factory(runtime_args)
    method = method_factory(args.method, model, runtime_args)
    records: list[dict[str, Any]] = []
    for batch in _chunks(items, int(args.generate_bs)):
        start = clock()
        results = _call_method(method, runtime_args, batch)
        elapsed = max(float(clock() - start), 0.0)
        per_item_latency = elapsed / max(len(results), 1)
        for item, result in zip(batch, results):
            records.append(_record_from_result(item=item, result=result, args=args, latency_sec=per_item_latency))

    _write_jsonl(args.prediction_output, records)
    summary = _summarize(args, records, config)
    meta_path = args.prediction_output.with_suffix(".jsonl.meta.json")
    meta_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latentmas-root", type=pathlib.Path, default=pathlib.Path("references/repos/LatentMAS"))
    parser.add_argument("--model-name", "--model_name", dest="model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True)
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--latentmas-task", default=None)
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--eval-file", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prediction-output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device2", default="cuda:1")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=256)
    parser.add_argument("--latent-steps", "--latent_steps", dest="latent_steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--generate-bs", "--generate_bs", dest="generate_bs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-vllm", "--use_vllm", dest="use_vllm", action="store_true")
    parser.add_argument("--enable-prefix-caching", "--enable_prefix_caching", dest="enable_prefix_caching", action="store_true")
    parser.add_argument("--use-second-hf-model", "--use_second_HF_model", dest="use_second_HF_model", action="store_true")
    parser.add_argument("--latent-space-realign", "--latent_space_realign", dest="latent_space_realign", action="store_true")
    parser.add_argument("--tensor-parallel-size", "--tensor_parallel_size", dest="tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", "--gpu_memory_utilization", dest="gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--text-mas-context-length", "--text_mas_context_length", dest="text_mas_context_length", type=int, default=-1)
    parser.add_argument("--think", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True
    return run_eval(args)


if __name__ == "__main__":
    main()
