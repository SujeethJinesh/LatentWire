from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_hidden_repair_packet_llm as llm_gate


def _chat_completions_url(api_base: str) -> str:
    stripped = api_base.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/v1/chat/completions"


def _completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(first.get("text"), str):
        return str(first["text"])
    return ""


def _completion_tokens(payload: dict[str, Any], generated_text: str) -> int:
    usage = payload.get("usage") or {}
    if isinstance(usage.get("completion_tokens"), int):
        return int(usage["completion_tokens"])
    return len(generated_text.encode("utf-8"))


def _post_chat_completion(
    *,
    api_base: str,
    api_key: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    seed: int,
) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    encoded = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(_chat_completions_url(api_base), data=encoded, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"endpoint HTTP {exc.code}: {detail}") from exc
    generated = _completion_text(payload).strip()
    return generated, _completion_tokens(payload, generated), payload, body


def _get_endpoint_models(*, api_base: str, api_key: str | None, timeout_s: float) -> dict[str, Any] | None:
    url = _chat_completions_url(api_base).removesuffix("/chat/completions") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _endpoint_metadata(
    *,
    api_base: str,
    api_key: str | None,
    timeout_s: float,
    served_model_id: str,
    served_model_revision: str,
    serving_engine: str,
    launch_command: str,
    dtype: str,
    quantization: str,
    tensor_parallel_size: str,
    max_model_len: str,
) -> dict[str, Any]:
    models_payload = _get_endpoint_models(api_base=api_base, api_key=api_key, timeout_s=timeout_s)
    return {
        "api_base_sha256": hashlib.sha256(api_base.encode("utf-8")).hexdigest(),
        "api_base_redacted": api_base,
        "models_endpoint_payload": models_payload,
        "served_model_id": served_model_id,
        "served_model_revision": served_model_revision,
        "serving_engine": serving_engine,
        "launch_command": launch_command,
        "dtype": dtype,
        "quantization": quantization,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
    }


def _generate_endpoint_packets(
    examples: list[llm_gate.LoadedExample],
    *,
    api_base: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    prompt_mode: str,
    timeout_s: float,
    seed: int,
    request_interval_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packets: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        prompt = llm_gate._prompt_for_diag(example, prompt_mode=prompt_mode)
        start = time.perf_counter()
        generated, completion_tokens, raw_payload, request_body = _post_chat_completion(
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            seed=seed,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        packet = llm_gate._extract_diag(generated)
        packets.append(
            {
                "example_id": example.example_id,
                "generated_text": generated,
                "latency_ms": latency_ms,
                "packet": packet,
                "packet_bytes": len(packet.encode("utf-8")),
                "packet_tokens": completion_tokens,
                "prompt_mode": prompt_mode,
                "valid_packet": bool(llm_gate.re.fullmatch(r"[A-Z][0-9]", packet)),
            }
        )
        trace.append(
            {
                "example_id": example.example_id,
                "index": index,
                "latency_ms": latency_ms,
                "request_body": request_body,
                "raw_response": raw_payload,
            }
        )
        if request_interval_s > 0 and index + 1 < len(examples):
            time.sleep(request_interval_s)
    return packets, trace


def _sha256_file(path: pathlib.Path) -> str:
    return llm_gate._sha256_file(path)


def _write_manifest_markdown(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# Source-Private Hidden-Repair Endpoint-Packet Manifest",
        "",
        "## Command",
        "",
        "```bash",
        manifest["command"],
        "```",
        "",
        "## Outcome",
        "",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- examples: `{summary['n']}`",
        f"- packet valid rate: `{summary['packet_valid_rate']:.3f}`",
        f"- matched model packet accuracy: `{summary['metrics']['matched_model_packet']['accuracy']:.3f}`",
        f"- target-only accuracy: `{summary['metrics']['target_only']['accuracy']:.3f}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{artifact}`" for artifact in manifest["artifacts"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-jsonl", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt-mode", choices=llm_gate.PROMPT_MODES, default="trace_no_hint")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--request-interval-s", type=float, default=0.0)
    parser.add_argument("--served-model-id", default=os.environ.get("SERVED_MODEL_ID", ""))
    parser.add_argument("--served-model-revision", default=os.environ.get("SERVED_MODEL_REVISION", ""))
    parser.add_argument("--serving-engine", default=os.environ.get("SERVED_ENGINE", ""))
    parser.add_argument("--launch-command", default=os.environ.get("SERVED_MODEL_LAUNCH_COMMAND", ""))
    parser.add_argument("--served-dtype", default=os.environ.get("SERVED_MODEL_DTYPE", ""))
    parser.add_argument("--served-quantization", default=os.environ.get("SERVED_MODEL_QUANTIZATION", ""))
    parser.add_argument("--tensor-parallel-size", default=os.environ.get("SERVED_MODEL_TENSOR_PARALLEL_SIZE", ""))
    parser.add_argument("--max-model-len", default=os.environ.get("SERVED_MODEL_MAX_MODEL_LEN", ""))
    args = parser.parse_args()

    benchmark_path = args.benchmark_jsonl if args.benchmark_jsonl.is_absolute() else ROOT / args.benchmark_jsonl
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env)
    endpoint_metadata = _endpoint_metadata(
        api_base=args.api_base,
        api_key=api_key,
        timeout_s=min(args.timeout_s, 10.0),
        served_model_id=args.served_model_id,
        served_model_revision=args.served_model_revision,
        serving_engine=args.serving_engine,
        launch_command=args.launch_command,
        dtype=args.served_dtype,
        quantization=args.served_quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    examples = llm_gate._load_examples(benchmark_path, limit=args.limit)
    packets, trace = _generate_endpoint_packets(
        examples,
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        prompt_mode=args.prompt_mode,
        timeout_s=args.timeout_s,
        seed=args.seed,
        request_interval_s=args.request_interval_s,
    )
    rows, summary = llm_gate._evaluate(examples, packets, seed=args.seed)

    llm_gate._write_jsonl(output_dir / "model_packets.jsonl", packets)
    llm_gate._write_jsonl(output_dir / "predictions.jsonl", rows)
    llm_gate._write_jsonl(output_dir / "endpoint_trace.jsonl", trace)
    (output_dir / "endpoint_metadata.json").write_text(
        json.dumps(endpoint_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    llm_gate._write_markdown(output_dir / "summary.md", summary)
    artifacts = [
        "model_packets.jsonl",
        "predictions.jsonl",
        "endpoint_trace.jsonl",
        "endpoint_metadata.json",
        "summary.json",
        "summary.md",
        "manifest.json",
        "manifest.md",
    ]
    manifest = {
        "command": " ".join(
            [
                "./venv_arm64/bin/python",
                "scripts/run_source_private_hidden_repair_packet_endpoint.py",
                f"--benchmark-jsonl {args.benchmark_jsonl}",
                f"--output-dir {args.output_dir}",
                f"--model {args.model}",
                f"--api-base {args.api_base}",
                f"--limit {args.limit}",
                f"--seed {args.seed}",
                f"--max-tokens {args.max_tokens}",
                f"--prompt-mode {args.prompt_mode}",
            ]
        ),
        "args": vars(args) | {"benchmark_jsonl": str(args.benchmark_jsonl), "output_dir": str(args.output_dir)},
        "endpoint_metadata": endpoint_metadata,
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "benchmark_sha256": _sha256_file(benchmark_path),
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest_markdown(output_dir / "manifest.md", manifest)
    if not summary["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
