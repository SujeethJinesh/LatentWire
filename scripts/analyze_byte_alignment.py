from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge import calibrate


DEFAULT_BYTE_STRESS_PROMPTS = [
    "Compute 7½% of $1,234.56, then add 3 km in meters.",
    "If a café sells 12 croissants at €2.50 each, what is the revenue?",
    "Solve: α + β = 17, and β = 5. What is α?",
    "Emoji check: 🧪 + 🧠 = one combined clue. What two objects are shown?",
    "Chemistry tokenization: NaCl, H₂O, CO₂, and 10⁻³ mol/L.",
    "Code stress: for i in range(3): total += nums[i]. What is loop count?",
    "Multilingual stress: 東京 to Paris is written as Tokyo to Paris in English.",
    "Units stress: 5 µs + 20 ms + 3 ns; which unit is largest?",
]


def _read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_BYTE_STRESS_PROMPTS)
    prompts: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            prompts.append(stripped)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _short_pairs(pairs: list[tuple[int, int]], limit: int) -> list[list[int]]:
    return [[int(src), int(tgt)] for src, tgt in pairs[:limit]]


def _build_record(
    *,
    prompt: str,
    src_tokenizer: Any,
    tgt_tokenizer: Any,
    max_length: int,
    source_reasoning_mode: str,
    source_use_chat_template: bool,
    source_enable_thinking: bool | None,
    target_use_chat_template: bool,
    target_enable_thinking: bool | None,
    pair_preview_limit: int,
) -> dict[str, Any]:
    src_text = calibrate._prepare_prompt_text(
        prompt,
        reasoning_mode=source_reasoning_mode,
        tokenizer=src_tokenizer,
        use_chat_template=source_use_chat_template,
        enable_thinking=source_enable_thinking,
    )
    tgt_text = calibrate._format_prompt_for_tokenizer(
        tgt_tokenizer,
        prompt,
        use_chat_template=target_use_chat_template,
        enable_thinking=target_enable_thinking,
    )
    span_pairs = calibrate.collect_aligned_prompt_position_pairs(
        src_tokenizer,
        tgt_tokenizer,
        [prompt],
        max_length=max_length,
        batch_size=1,
        source_reasoning_mode=source_reasoning_mode,
        source_use_chat_template=source_use_chat_template,
        source_enable_thinking=source_enable_thinking,
        target_use_chat_template=target_use_chat_template,
        target_enable_thinking=target_enable_thinking,
    )[0]
    byte_pairs = calibrate.collect_byte_aligned_prompt_position_pairs(
        src_tokenizer,
        tgt_tokenizer,
        [prompt],
        max_length=max_length,
        batch_size=1,
        source_reasoning_mode=source_reasoning_mode,
        source_use_chat_template=source_use_chat_template,
        source_enable_thinking=source_enable_thinking,
        target_use_chat_template=target_use_chat_template,
        target_enable_thinking=target_enable_thinking,
    )[0]
    src_len = calibrate._prompt_valid_length_for_text(src_tokenizer, src_text, max_length=max_length)
    tgt_len = calibrate._prompt_valid_length_for_text(tgt_tokenizer, tgt_text, max_length=max_length)
    span_set = set(span_pairs)
    byte_set = set(byte_pairs)
    return {
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "source_tokens": src_len,
        "target_tokens": tgt_len,
        "span_pair_count": len(span_pairs),
        "byte_pair_count": len(byte_pairs),
        "changed": span_pairs != byte_pairs,
        "only_span_pair_count": len(span_set - byte_set),
        "only_byte_pair_count": len(byte_set - span_set),
        "span_pairs_preview": _short_pairs(span_pairs, pair_preview_limit),
        "byte_pairs_preview": _short_pairs(byte_pairs, pair_preview_limit),
    }


def _write_markdown(records: list[dict[str, Any]], path: Path) -> None:
    changed = [record for record in records if record["changed"]]
    lines = [
        "# Byte Alignment Audit",
        "",
        f"- prompts: `{len(records)}`",
        f"- changed prompts: `{len(changed)}`",
        f"- mean span pairs: `{mean(record['span_pair_count'] for record in records):.2f}`",
        f"- mean byte pairs: `{mean(record['byte_pair_count'] for record in records):.2f}`",
        "",
        "| Changed | UTF-8 bytes | Src toks | Tgt toks | Span pairs | Byte pairs | Prompt |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        prompt = str(record["prompt"]).replace("|", "\\|")
        lines.append(
            f"| {int(bool(record['changed']))} | {record['prompt_utf8_bytes']} | "
            f"{record['source_tokens']} | {record['target_tokens']} | "
            f"{record['span_pair_count']} | {record['byte_pair_count']} | {prompt} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether byte-dominant token alignment differs from char-span alignment."
    )
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--source-reasoning-mode", default="plain")
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--target-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", choices=["true", "false", "none"], default="none")
    parser.add_argument("--target-enable-thinking", choices=["true", "false", "none"], default="none")
    parser.add_argument("--pair-preview-limit", type=int, default=32)
    return parser.parse_args()


def _thinking_value(raw: str) -> bool | None:
    if raw == "true":
        return True
    if raw == "false":
        return False
    return None


def main() -> None:
    args = parse_args()
    prompts = _read_prompts(args.prompt_file)
    src_tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    src_enable_thinking = _thinking_value(args.source_enable_thinking)
    tgt_enable_thinking = _thinking_value(args.target_enable_thinking)

    records = [
        _build_record(
            prompt=prompt,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_length=args.max_length,
            source_reasoning_mode=args.source_reasoning_mode,
            source_use_chat_template=args.source_use_chat_template,
            source_enable_thinking=src_enable_thinking,
            target_use_chat_template=args.target_use_chat_template,
            target_enable_thinking=tgt_enable_thinking,
            pair_preview_limit=args.pair_preview_limit,
        )
        for prompt in prompts
    ]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    _write_markdown(records, args.output_md)

    changed = sum(int(record["changed"]) for record in records)
    print(
        "Byte alignment audit: "
        f"prompts={len(records)}, changed={changed}, "
        f"mean_span_pairs={mean(record['span_pair_count'] for record in records):.2f}, "
        f"mean_byte_pairs={mean(record['byte_pair_count'] for record in records):.2f}"
    )


if __name__ == "__main__":
    main()
