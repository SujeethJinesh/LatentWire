"""Fixed-request vLLM profiling driver for HybridKernel.

This script is meant for a native NVIDIA host running a vLLM OpenAI-compatible
server.  It is deliberately small and dependency-free so the profiler runbook
has an executable driver instead of an ad hoc curl loop.  On Mac, use
`--dry-run` to verify the request matrix without needing vLLM or a GPU.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RequestRow:
    request_id: int
    batch_size: int
    prefill_tokens: int
    decode_tokens: int
    elapsed_s: float | None
    status: str


def _prompt(prefill_tokens: int, request_id: int) -> str:
    # Approximate a fixed-token prompt without depending on a tokenizer.
    words = [f"tok{(request_id + idx) % 997}" for idx in range(prefill_tokens)]
    return " ".join(words)


def _payload(model: str, prompt: str | list[str], decode_tokens: int, seed: int) -> dict[str, object]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": decode_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
    }


def _post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}")
        response.read()


def run(args: argparse.Namespace) -> dict[str, object]:
    base_endpoint = args.endpoint.rstrip("/")
    endpoint = base_endpoint + "/v1/completions"
    profile_start_endpoint = base_endpoint + "/start_profile"
    profile_stop_endpoint = base_endpoint + "/stop_profile"
    rows: list[RequestRow] = []

    if args.profile_bracket and not args.dry_run:
        _post_json(profile_start_endpoint, {}, args.timeout_s)

    try:
        for request_id in range(args.requests):
            prompts = [
                _prompt(args.prefill_tokens, request_id * args.batch_size + batch_idx)
                for batch_idx in range(args.batch_size)
            ]
            prompt_payload: str | list[str] = prompts[0] if args.batch_size == 1 else prompts
            payload = _payload(args.model, prompt_payload, args.decode_tokens, args.seed + request_id)
            if args.dry_run:
                rows.append(
                    RequestRow(
                        request_id=request_id,
                        batch_size=args.batch_size,
                        prefill_tokens=args.prefill_tokens,
                        decode_tokens=args.decode_tokens,
                        elapsed_s=None,
                        status="dry_run",
                    )
                )
                continue
            start = time.perf_counter()
            try:
                _post_json(endpoint, payload, args.timeout_s)
                status = "ok"
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                status = f"error:{exc}"
            rows.append(
                RequestRow(
                    request_id=request_id,
                    batch_size=args.batch_size,
                    prefill_tokens=args.prefill_tokens,
                    decode_tokens=args.decode_tokens,
                    elapsed_s=time.perf_counter() - start,
                    status=status,
                )
            )
    finally:
        if args.profile_bracket and not args.dry_run:
            _post_json(profile_stop_endpoint, {}, args.timeout_s)

    return {
        "model": args.model,
        "endpoint": endpoint,
        "profile_bracket": args.profile_bracket,
        "profile_start_endpoint": profile_start_endpoint if args.profile_bracket else None,
        "profile_stop_endpoint": profile_stop_endpoint if args.profile_bracket else None,
        "dry_run": args.dry_run,
        "requests": [asdict(row) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--profile-bracket",
        action="store_true",
        help="POST /start_profile before fixed requests and /stop_profile after them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
