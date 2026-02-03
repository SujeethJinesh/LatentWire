"""Text-only communication baselines (byte-capped)."""
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional


def count_utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def truncate_utf8(text: str, budget_bytes: int) -> Tuple[str, bool]:
    if budget_bytes is None:
        return text, False
    if budget_bytes <= 0:
        return "", True
    data = text.encode("utf-8")
    if len(data) <= budget_bytes:
        return text, False
    truncated = data[:budget_bytes].decode("utf-8", errors="ignore")
    return truncated, True


def _extract_question_choices(example: Dict[str, Any]) -> Tuple[str, str]:
    question = example.get("question_stem") or example.get("question") or example.get("query") or ""
    choices = example.get("choices") or {}
    if isinstance(choices, dict) and "text" in choices:
        choices_list = choices.get("text") or []
    elif isinstance(choices, list):
        choices_list = choices
    else:
        choices_list = []
    if not choices_list:
        # Fallback to answerKey choices if present
        for key in ("options", "answers"):
            if key in example and isinstance(example[key], list):
                choices_list = example[key]
                break
    formatted = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices_list)])
    return question, formatted


def make_message(example: Dict[str, Any], budget_bytes: int, style: str,
                 llm_generate_fn: Optional[Any] = None) -> Dict[str, Any]:
    style = (style or "text_raw").lower()
    question, choices = _extract_question_choices(example)
    base = f"Question:\n{question}\n\nChoices:\n{choices}\n"

    if style == "text_raw":
        msg = base
    elif style == "text_summary_heur":
        # Simple heuristic: collapse whitespace and truncate aggressively.
        msg = " ".join(base.split())
    elif style == "text_summary_llm":
        if llm_generate_fn is None:
            msg = " ".join(base.split())
        else:
            msg = llm_generate_fn(question, choices, budget_bytes)
    else:
        msg = base

    msg, truncated = truncate_utf8(msg, budget_bytes)
    return {
        "message": msg,
        "bytes": count_utf8_bytes(msg),
        "truncated": truncated,
        "budget_bytes": budget_bytes,
    }
