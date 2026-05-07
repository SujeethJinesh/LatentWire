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
from typing import Any


@dataclass(frozen=True)
class RequestRow:
    request_id: int
    batch_size: int
    prefill_tokens: int
    decode_tokens: int
    prompt_token_counts: list[int] | None
    prompt_token_count_total: int | None
    requested_decode_tokens: int
    expected_completion_tokens_total: int
    response_usage: dict[str, Any] | None
    elapsed_s: float | None
    status: str


def _prompt(prefill_tokens: int, request_id: int) -> str:
    # Approximate a fixed-token prompt without depending on a tokenizer.
    words = [f"tok{(request_id + idx) % 997}" for idx in range(prefill_tokens)]
    return " ".join(words)


def _tokenizer_vocab_size(tokenizer: Any) -> int | None:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    try:
        length = len(tokenizer)
    except TypeError:
        return None
    return length if isinstance(length, int) and length > 0 else None


def _special_token_ids(tokenizer: Any) -> set[int]:
    raw_ids = getattr(tokenizer, "all_special_ids", []) or []
    return {int(token_id) for token_id in raw_ids if isinstance(token_id, int)}


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str | None:
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None
    try:
        decoded = decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = decode(token_ids)
    return decoded if isinstance(decoded, str) and decoded else None


def _count_tokens(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    return len(encoded)


def _exact_prompt_from_tokenizer(tokenizer: Any, prefill_tokens: int, request_id: int) -> str | None:
    """Synthesize a prompt that round-trips to exactly ``prefill_tokens`` tokens.

    The profiler gate compares latency/HBM statistics at specific prefill sizes,
    so approximate whitespace prompts are too weak when ``--require-token-counts``
    is enabled.  Most Hugging Face tokenizers can decode ordinary token ids back
    into text that re-encodes to the same token count; if that invariant does not
    hold locally, the caller falls back to the existing explicit count mismatch.
    """

    if prefill_tokens < 0:
        raise ValueError("--prefill-tokens must be non-negative")
    if prefill_tokens == 0:
        return ""

    vocab_size = _tokenizer_vocab_size(tokenizer)
    if vocab_size is None:
        return None
    special_ids = _special_token_ids(tokenizer)
    candidate_ids = [token_id for token_id in range(vocab_size) if token_id not in special_ids]
    if not candidate_ids:
        return None

    start = (request_id * 131) % len(candidate_ids)
    single_token_tries = min(64, len(candidate_ids))
    for offset in range(single_token_tries):
        token_id = candidate_ids[(start + offset) % len(candidate_ids)]
        prompt = _decode_token_ids(tokenizer, [token_id] * prefill_tokens)
        if prompt is not None and _count_tokens(tokenizer, prompt) == prefill_tokens:
            return prompt

    stride_tries = (1, 3, 7, 13)
    for stride in stride_tries:
        ids = [candidate_ids[(start + idx * stride) % len(candidate_ids)] for idx in range(prefill_tokens)]
        prompt = _decode_token_ids(tokenizer, ids)
        if prompt is not None and _count_tokens(tokenizer, prompt) == prefill_tokens:
            return prompt

    return None


def _payload(model: str, prompt: str | list[str], decode_tokens: int, seed: int) -> dict[str, object]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": decode_tokens,
        "min_tokens": decode_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
    }


def _post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> dict[str, Any] | None:
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
        response_body = response.read()
    if not response_body:
        return None
    try:
        parsed = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _load_tokenizer(model: str, tokenizer_name: str | None, require: bool) -> tuple[Any | None, str | None, str]:
    name = tokenizer_name or (model if require else None)
    if name is None:
        return None, None, "not_requested"
    try:
        from transformers import AutoTokenizer  # type: ignore

        return (
            AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=True),
            name,
            "transformers_local_files_only",
        )
    except Exception as exc:  # pragma: no cover - exact dependency/cache errors are environment-specific.
        if require:
            raise RuntimeError(
                f"token counting was required but tokenizer {name!r} could not be loaded locally: {exc}"
            ) from exc
        return None, name, f"unavailable:{exc}"


def _prompt_token_counts(tokenizer: Any | None, prompt_payload: str | list[str]) -> list[int] | None:
    if tokenizer is None:
        return None
    prompts = [prompt_payload] if isinstance(prompt_payload, str) else prompt_payload
    return [_count_tokens(tokenizer, prompt) for prompt in prompts]


def _planned_requests(
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> list[tuple[int, str | list[str], list[int] | None, dict[str, object]]]:
    planned = []
    enforce_counts = bool(getattr(args, "enforce_prefill_token_counts", False)) or bool(
        getattr(args, "require_token_counts", False)
    )
    for request_id in range(args.requests):
        prompts = [
            _exact_prompt_from_tokenizer(tokenizer, args.prefill_tokens, request_id * args.batch_size + batch_idx)
            if enforce_counts and tokenizer is not None
            else None
            for batch_idx in range(args.batch_size)
        ]
        prompts = [
            prompt
            if prompt is not None
            else _prompt(args.prefill_tokens, request_id * args.batch_size + batch_idx)
            for batch_idx, prompt in enumerate(prompts)
        ]
        prompt_payload: str | list[str] = prompts[0] if args.batch_size == 1 else prompts
        prompt_token_counts = _prompt_token_counts(tokenizer, prompt_payload)
        if enforce_counts:
            if prompt_token_counts is None:
                raise RuntimeError("prefill token-count enforcement requires a loaded tokenizer")
            mismatched = [count for count in prompt_token_counts if count != args.prefill_tokens]
            if mismatched:
                raise ValueError(
                    "prompt token counts do not match --prefill-tokens="
                    f"{args.prefill_tokens}: {prompt_token_counts}"
                )
        payload = _payload(args.model, prompt_payload, args.decode_tokens, args.seed + request_id)
        planned.append((request_id, prompt_payload, prompt_token_counts, payload))
    return planned


def run(args: argparse.Namespace) -> dict[str, object]:
    base_endpoint = args.endpoint.rstrip("/")
    endpoint = base_endpoint + "/v1/completions"
    profile_start_endpoint = base_endpoint + "/start_profile"
    profile_stop_endpoint = base_endpoint + "/stop_profile"
    rows: list[RequestRow] = []
    tokenizer, tokenizer_name, token_count_source = _load_tokenizer(
        args.model,
        getattr(args, "tokenizer", None),
        bool(getattr(args, "require_token_counts", False)),
    )
    planned_requests = _planned_requests(args, tokenizer)

    if args.profile_bracket and not args.dry_run:
        _post_json(profile_start_endpoint, {}, args.timeout_s)

    try:
        for request_id, _prompt_payload, prompt_token_counts, payload in planned_requests:
            if args.dry_run:
                rows.append(
                    RequestRow(
                        request_id=request_id,
                        batch_size=args.batch_size,
                        prefill_tokens=args.prefill_tokens,
                        decode_tokens=args.decode_tokens,
                        prompt_token_counts=prompt_token_counts,
                        prompt_token_count_total=sum(prompt_token_counts) if prompt_token_counts else None,
                        requested_decode_tokens=args.decode_tokens,
                        expected_completion_tokens_total=args.batch_size * args.decode_tokens,
                        response_usage=None,
                        elapsed_s=None,
                        status="dry_run",
                    )
                )
                continue
            start = time.perf_counter()
            try:
                response = _post_json(endpoint, payload, args.timeout_s)
                status = "ok"
                response_usage = response.get("usage") if isinstance(response, dict) else None
                if not isinstance(response_usage, dict):
                    response_usage = None
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                status = f"error:{exc}"
                response_usage = None
            rows.append(
                RequestRow(
                    request_id=request_id,
                    batch_size=args.batch_size,
                    prefill_tokens=args.prefill_tokens,
                    decode_tokens=args.decode_tokens,
                    prompt_token_counts=prompt_token_counts,
                    prompt_token_count_total=sum(prompt_token_counts) if prompt_token_counts else None,
                    requested_decode_tokens=args.decode_tokens,
                    expected_completion_tokens_total=args.batch_size * args.decode_tokens,
                    response_usage=response_usage,
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
        "tokenizer": tokenizer_name,
        "token_count_source": token_count_source,
        "token_counts_required": bool(getattr(args, "require_token_counts", False)),
        "prefill_token_counts_enforced": bool(getattr(args, "enforce_prefill_token_counts", False))
        or bool(getattr(args, "require_token_counts", False)),
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
    parser.add_argument(
        "--tokenizer",
        help="Optional local tokenizer id/path for exact prompt token-count logging.",
    )
    parser.add_argument(
        "--require-token-counts",
        action="store_true",
        help="Fail unless the tokenizer can be loaded locally and prompt token counts are logged.",
    )
    parser.add_argument(
        "--enforce-prefill-token-counts",
        action="store_true",
        help="Fail before profiling unless every prompt tokenizes to --prefill-tokens.",
    )
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
